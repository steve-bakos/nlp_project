def compute_realignment_loss_for_adapter(
    model,
    realignment_transformation,
    strong_alignment=True,
    realignment_temperature=0.1,
    realignment_loss="contrastive",
    # sentences from left language
    left_input_ids=None,
    left_attention_mask=None,
    left_token_type_ids=None,
    left_position_ids=None,
    left_head_mask=None,
    left_inputs_embeds=None,
    left_lang_id=None,
    # sentences from right language
    right_input_ids=None,
    right_attention_mask=None,
    right_token_type_ids=None,
    right_position_ids=None,
    right_head_mask=None,
    right_inputs_embeds=None,
    right_lang_id=None,
    # alignment labels
    alignment_left_ids=None,  # [word_id] -> [[word_id | -1]]
    alignment_left_positions=None,  # [[batch, start, end]] -> [[[start, end]]]
    alignment_right_ids=None,  # [word_id]
    alignment_right_positions=None,  # [[batch, start, end]]
    alignment_nb=None,  # [[i]]
    alignment_left_length=None,  # [[i]]
    alignment_right_length=None,  # [[i]]
):
    # TODO build new batch with mapping -> Do it in a specific collator

    # TODO use BatchSplit and pass to model

    # TODO compute the loss

    pass
