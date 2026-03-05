import torch


def expand_vocab_with_interpolation(pretrained_state_dict, new_config, tokenizer, original_decode_vocab=8487):
    """
    Expand decoder_embedding and lm_head from original_decode_vocab -> new_config.decode_vocab_size.
    First original_decode_vocab rows: direct copy from pretrained.
    Rows original_decode_vocab+: interpolated from flanking western semitone embeddings.

    Args:
        pretrained_state_dict: state dict from pretrained checkpoint (original 8487 vocab)
        new_config: LlamaConfig with expanded decode_vocab_size (e.g., 9675)
        tokenizer: MusicTokenizer instance with microtonal pitch_dict (append-only layout)
        original_decode_vocab: original decode_vocab_size (default 8487)

    Returns:
        new state dict with expanded decoder_embedding and lm_head weights
    """
    reverse_pitch_dict = {v: k for k, v in tokenizer.pitch_dict.items()}

    # Row offset where western pitch classes (0-11) start in the flat GRU vocab
    pitch_offset = (tokenizer.sos_out_vocab_size + tokenizer.timeshift_vocab_size +
                    tokenizer.dur_vocab_size + tokenizer.octave_vocab_size)

    new_vocab_size = new_config.decode_vocab_size

    def _interpolate_new_rows(old_weight):
        """Expand a (original_decode_vocab, D) weight matrix to (new_vocab_size, D).

        Rows 0..original_decode_vocab-1 are copied verbatim. New rows (appended
        microtonal pitch IDs) are linearly interpolated from the two flanking
        western semitone embeddings. For example, cent 50 (halfway between C=0
        and C#=100) gets 0.5*E[C] + 0.5*E[C#].
        """
        new_weight = torch.zeros(new_vocab_size, old_weight.shape[1], dtype=old_weight.dtype)
        new_weight[:original_decode_vocab] = old_weight

        for new_id in range(original_decode_vocab, new_vocab_size):
            if new_id not in reverse_pitch_dict:
                continue
            cent = reverse_pitch_dict[new_id]
            lower_pc = cent // 100              # 0-11, western semitone below
            upper_pc = (lower_pc + 1) % 12      # wraps B(11)->C(0) at octave boundary
            frac = (cent % 100) / 100.0          # interpolation fraction
            lower_row = old_weight[pitch_offset + lower_pc]
            upper_row = old_weight[pitch_offset + upper_pc]
            new_weight[new_id] = (1 - frac) * lower_row + frac * upper_row

        return new_weight

    new_state = {}
    for key, value in pretrained_state_dict.items():
        if 'decoder_embedding.weight' in key:
            new_state[key] = _interpolate_new_rows(value)
        elif 'lm_head.weight' in key:
            new_state[key] = _interpolate_new_rows(value)
        else:
            new_state[key] = value  # All other params: direct copy

    return new_state
