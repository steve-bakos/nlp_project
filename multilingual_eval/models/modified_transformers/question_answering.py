from typing import Optional, List, Union, Tuple
import torch
from transformers import (
    BertForQuestionAnswering,
    XLMRobertaForQuestionAnswering,
    BertModel,
    XLMRobertaModel,
    DistilBertModel,
    DistilBertForQuestionAnswering,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from multilingual_eval.models.modified_transformers.bert_model import (
    encoder_with_optional_mapping_factory,
)
from multilingual_eval.models.modified_transformers.utils import get_class_from_model_path


def question_answering_with_optional_mapping_factory(BaseClass):
    """
    Factory function for creating a custom class for a sequence classification model
    that builds on top of an existing one (BaseClass) by adding an optional orthogonal
    mapping to the encoder (and lang_id in the forward method)
    """

    class CustomModelForQuestionAnswering(BaseClass):
        """
        Custom class for a sequence classification model
        that builds on top of an existing one (BaseClass) by adding an optional orthogonal
        mapping to the encoder (and lang_id in the forward method)
        """

        def __init__(self, config, with_mapping=False, nb_pairs=1):
            super().__init__(config)

            self.with_mapping = with_mapping

            encoder_class = getattr(self, BaseClass.base_model_prefix).__class__

            if with_mapping:
                setattr(
                    self,
                    BaseClass.base_model_prefix,
                    encoder_with_optional_mapping_factory(encoder_class)(
                        config,
                        add_pooling_layer=False,
                        with_mapping=with_mapping,
                        nb_pairs=nb_pairs,
                    ),
                )

            self.post_init()

        def get_encoder(self):
            """
            Allows to get the encoder directly
            """
            return getattr(self, BaseClass.base_model_prefix)

        def orthogonalize(self):
            """
            Orthogonalize the mapping (throws an error if there is not any)
            """
            getattr(self, BaseClass.base_model_prefix).mapping.orthogonalize()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            lang_id: Optional[torch.Tensor] = None,
            train_only_mapping=False,
        ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
            """
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = getattr(self, BaseClass.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **(
                    {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
                ),  # This is because of distilbert
                **({"position_ids": position_ids} if position_ids is not None else {}),
                **(
                    {"lang_id": lang_id, "train_only_mapping": train_only_mapping}
                    if self.with_mapping
                    else {}
                ),  # Had to rewrite the forward method because of this line
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return CustomModelForQuestionAnswering


CustomBertForQuestionAnswering = question_answering_with_optional_mapping_factory(
    BertForQuestionAnswering,
)
CustomRobertaForQuestionAnswering = question_answering_with_optional_mapping_factory(
    XLMRobertaForQuestionAnswering,
)
CustomDistilBertForQuestionAnswering = question_answering_with_optional_mapping_factory(
    DistilBertForQuestionAnswering,
)


class CustomAutoModelForQuestionAnswering:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return question_answering_with_optional_mapping_factory(
                BertForQuestionAnswering,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return question_answering_with_optional_mapping_factory(
                XLMRobertaForQuestionAnswering,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return question_answering_with_optional_mapping_factory(
                DistilBertForQuestionAnswering,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"CustomAutoModelForQuestionAnswering.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )
