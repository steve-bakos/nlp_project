from typing import Optional, Union, Tuple, List
import torch
from transformers import BertModel, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from multilingual_eval.models.multiple_mapping import MultipleMappings
from multilingual_eval.models.contexts import DumbContext


def encoder_with_optional_mapping_factory(BaseClass):
    """
    Factory function for creating a custome encoder class that modifies an existing encoder class from
    the transformers library. This custom class allows to include an optional mapping
    for realignment, which necessitates a new argument in the forward method, lang_id, indicating
    which mapping to use (according to the language of the sentence passed)

    It takes as argument an encoder class like BertModel or RobertaModel
    """

    if not issubclass(BaseClass, (BertModel, XLMRobertaModel)):
        raise NotImplementedError(
            f"encoder_with_optional_mapping_factory only works with BertModel or XLMRobertaModel, got {BaseClass.__name__}"
        )

    class CustomEncoder(BaseClass):
        """
        Custom encoder class which includes an optional mapping or realignment,
        which necessitates a new argument in the forward method, lang_id, indicating
        which mapping to use (according to the language of the sentence passed)
        """

        def __init__(self, config, add_pooling_layer=True, with_mapping=False, nb_pairs=1):
            super().__init__(config, add_pooling_layer=add_pooling_layer)
            self.mapping = None
            self.real_pooler = None
            self.with_mapping = with_mapping
            self.nb_pairs = nb_pairs
            if with_mapping:
                self.real_pooler = self.pooler
                self.pooler = None

                self.mapping = MultipleMappings(nb_pairs, config.hidden_size)

        def train_mapping(self, value: bool):
            if self.mapping is not None:
                for param in self.mapping.parameters():
                    param.requires_grad = value

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            lang_id: Optional[torch.Tensor] = None,
            train_only_mapping=False,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

            context = torch.no_grad() if train_only_mapping else DumbContext()

            with context:
                res = super().forward(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    position_ids,
                    head_mask,
                    inputs_embeds,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict=True,
                )

            new_sequence_output = (
                self.mapping(res.last_hidden_state, lang_id)
                if self.mapping is not None
                else res.last_hidden_state
            )

            new_pooled_output = (
                self.real_pooler(new_sequence_output)
                if self.real_pooler is not None
                else res.pooler_output
            )

            new_all_hidden_states = (
                (*res.hidden_states[:-1], new_sequence_output)
                if res.hidden_states is not None
                else None
            )

            if not return_dict:
                return (new_sequence_output, new_pooled_output) + tuple(
                    v
                    for v in [
                        res.past_key_values,
                        new_all_hidden_states,
                        res.attentions,
                        res.cross_attentions,
                    ]
                    if v is not None
                )

            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=new_sequence_output,
                pooler_output=new_pooled_output,
                hidden_states=new_all_hidden_states,
                past_key_values=res.past_key_values,
                attentions=res.attentions,
                cross_attentions=res.cross_attentions,
            )

    return CustomEncoder
