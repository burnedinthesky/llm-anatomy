import torch
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path


def rms_norm(
    tensor: torch.Tensor, weight: torch.Tensor, norm_eps: float
) -> torch.Tensor:
    pass


def rope_rotate(embeddings: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # Split into pairs for RoPE
    embedding_split = embeddings.float().view(embeddings.shape[0], -1, 2)

    # Convert to complex numbers
    embedding_complex = torch.view_as_complex(embedding_split)

    # Apply RoPE rotation
    embedding_rotated_complex = embedding_complex * freqs_cis

    # Convert back to real
    embedding_rotated = torch.view_as_real(embedding_rotated_complex).view(
        embedding_split.shape
    )

    return embedding_rotated


def init_tokenizer(tokenizer_path: str) -> tiktoken.Encoding:
    # Initialize tokenizer with the pretrained model
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name=Path(tokenizer_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={
            token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)
        },
    )
    return tokenizer
