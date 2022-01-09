import torch


def pad_tensor(sequence: torch.Tensor,
               max_seq_len: int,
               pad_value: float = 0.) -> torch.Tensor:
    """
    Pads an embedded sequence of tokens of shape (n_tokens, d_embedding) on the first dimension.
    The resulting tensor is of shape (max_seq_len, d_embedding)

    :param sequence:
    :type sequence:
    :param max_seq_len:
    :type max_seq_len:
    :param pad_value:
    :type pad_value:
    :return:
    :rtype:
    """
    # Replace multiple spaces with single space.

    seq_len, d_embedding = sequence.shape

    padding = torch.full((d_embedding,), fill_value=pad_value, dtype=torch.float32)

    diff = max_seq_len - seq_len

    assert diff > 0, \
        f"Length of sequence '{sequence}' is greater than the max_length - {seq_len} > {max_seq_len}."

    if diff % 2 == 0:
        left_pad_len = diff // 2
        right_pad_len = diff // 2
    else:
        left_pad_len = diff // 2
        right_pad_len = diff // 2 + 1

    assert left_pad_len + right_pad_len + seq_len == max_seq_len, \
        f'Sum of {left_pad_len}, {right_pad_len}, and {diff} != {max_seq_len}'

    # left padding (left_pad_len, max_seq_len)
    left_padding = torch.vstack([padding for _ in range(left_pad_len)])

    # right_padding (right_pad_len, max_seq_len)
    right_padding = torch.vstack([padding for _ in range(right_pad_len)])

    padded_sequence = torch.cat((left_padding, sequence, right_padding), dim=0)

    del left_padding
    del right_padding

    assert len(padded_sequence) == max_seq_len, \
        f'Final length ({len(padded_sequence)}) of padded sequence {padded_sequence} is different from {max_len}.'

    return padded_sequence
