from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    freqs = torch.pow(theta, -torch.arange(0, head_dim, 2, device=device)[:(head_dim//2)].float() / head_dim)
    pos = torch.arange(seqlen, device=device).float()[:max_seq_len]

    freqs = torch.outer(freqs, pos).transpose(-2, -1).float()  # (head_dim // 2, max_seq_len)
    freqs = reshape_for_broadcast(freqs, query_real)

    # shape: (batch_size, seqlen, n_local_heads, head_dim // 2)
    query_rotated_real = freqs.cos() * query_real - freqs.sin() * query_imag
    query_rotated_imag = freqs.sin() * query_real + freqs.cos() * query_imag
    key_rotated_real = freqs.cos() * key_real - freqs.sin() * key_imag
    key_rotated_imag = freqs.sin() * key_real + freqs.cos() * key_imag

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_stack = torch.stack((query_rotated_real, query_rotated_imag), dim=-1)
    key_stack = torch.stack((key_rotated_real, key_rotated_imag), dim=-1)

    query_out = query_stack.reshape(query.shape)
    key_out = key_stack.reshape(key.shape)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out