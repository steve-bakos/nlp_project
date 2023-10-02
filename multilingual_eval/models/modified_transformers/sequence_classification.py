from typing import Optional, Union, Tuple
import torch
from transformers import (
    BertForSequenceClassification,
    XLMRobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    BertModel,
    XLMRobertaModel,
    DistilBertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from multilingual_eval.models.modified_transformers.bert_model import (
    encoder_with_optional_mapping_factory,
)
from multilingual_eval.models.modified_transformers.utils import get_class_from_model_path


def sequence_classifier_with_optional_mapping_factory(BaseClass, classifier_getter=None):
    """
    Factory function for creating a custom class for a token classification model
    that builds on top of an existing one (BaseClass) by adding an optional orthogonal
    mapping to the encoder (and lang_id in the forward method)
    """

    class CustomModelForSequenceClassification(BaseClass):
        """
        Custom class for a token classification model
        that builds on top of an existing one (BaseClass) by adding an optional orthogonal
        mapping to the encoder (and lang_id in the forward method)
        """

        def __init__(self, config, with_mapping=False, nb_pairs=1):
            super().__init__(config)

            self.with_mapping = with_mapping
            self.classifier_getter = classifier_getter

            encoder_class = getattr(self, BaseClass.base_model_prefix).__class__

            if with_mapping:
                setattr(
                    self,
                    BaseClass.base_model_prefix,
                    encoder_with_optional_mapping_factory(encoder_class)(
                        config, add_pooling_layer=True, with_mapping=with_mapping, nb_pairs=nb_pairs
                    ),
                )

            self.post_init()

        def get_encoder(self):
            return getattr(self, BaseClass.base_model_prefix)

        def orthogonalize(self):
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
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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

            if self.classifier_getter is None:
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
            else:
                logits = self.classifier_getter(self)(outputs)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = torch.nn.MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return CustomModelForSequenceClassification


def roberta_sequence_classifier_getter(model: XLMRobertaForSequenceClassification):
    def classifier(outputs):
        sequence_output = outputs[0]
        logits = model.classifier(sequence_output)

        return logits

    return classifier


def distilbert_sequence_classifier_getter(model: DistilBertForSequenceClassification):
    def classifier(outputs):
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = model.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = model.dropout(pooled_output)  # (bs, dim)
        logits = model.classifier(pooled_output)  # (bs, num_labels)
        return logits

    return classifier


CustomBertForSequenceClassification = sequence_classifier_with_optional_mapping_factory(
    BertForSequenceClassification
)
CustomRobertaForSequenceClassification = sequence_classifier_with_optional_mapping_factory(
    XLMRobertaForSequenceClassification, classifier_getter=roberta_sequence_classifier_getter
)
CustomDistilBertForSequenceClassification = sequence_classifier_with_optional_mapping_factory(
    DistilBertForSequenceClassification, classifier_getter=distilbert_sequence_classifier_getter
)


class CustomAutoModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, path: str, *args, cache_dir=None, **kwargs):
        model_class = get_class_from_model_path(path, cache_dir=cache_dir)

        if issubclass(model_class, BertModel):
            return sequence_classifier_with_optional_mapping_factory(
                BertForSequenceClassification
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, XLMRobertaModel):
            return sequence_classifier_with_optional_mapping_factory(
                XLMRobertaForSequenceClassification,
                classifier_getter=roberta_sequence_classifier_getter,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        elif issubclass(model_class, DistilBertModel):
            return sequence_classifier_with_optional_mapping_factory(
                DistilBertForSequenceClassification,
                classifier_getter=distilbert_sequence_classifier_getter,
            ).from_pretrained(path, *args, cache_dir=cache_dir, **kwargs)
        else:
            raise Exception(
                f"CustomAutoModelForSequenceClassification.from_pretrained is not compatible with model of class `{model_class}` (path: {path})"
            )
