from typing import Optional, List, Union, Tuple
import torch
from transformers import (
    BertForTokenClassification,
    XLMRobertaForTokenClassification,
    BertModel,
    XLMRobertaModel,
    DistilBertModel,
    DistilBertForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput

from multilingual_eval.models.modified_transformers.bert_model import (
    encoder_with_optional_mapping_factory,
)
from multilingual_eval.models.modified_transformers.utils import get_class_from_model_path


def token_classifier_with_optional_mapping_factory(BaseClass):
    """
    Factory function for creating a custom class for a sequence classification model
    that builds on top of an existing one (BaseClass) by adding an optional orthogonal
    mapping to the encoder (and lang_id in the forward method)
    """

    class CustomModelForTokenClassification(BaseClass):
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
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            lang_id: Optional[torch.Tensor] = None,
            train_only_mapping=False,
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
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

            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return CustomModelForTokenClassification


CustomBertForTokenClassification = token_classifier_with_optional_mapping_factory(
    BertForTokenClassification,
)
CustomRobertaForTokenClassification = token_classifier_with_optional_mapping_factory(
    XLMRobertaForTokenClassification,
)
CustomDistilBertForTokenClassification = token_classifier_with_optional_mapping_factory(
    DistilBertForTokenClassification,
)


class CustomAutoModelForTokenClassification:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return token_classifier_with_optional_mapping_factory(
                BertForTokenClassification,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return token_classifier_with_optional_mapping_factory(
                XLMRobertaForTokenClassification,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return token_classifier_with_optional_mapping_factory(
                DistilBertForTokenClassification,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"CustomAutoModelForTokenClassification.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )
