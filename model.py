import torch
import json
from os import path
import utils


class LlamaModel:
    def __init__(self, model_dir: str | None = None, bos_token: int = 128000):
        self.model_dir = model_dir
        self.model = None
        self.bos_token = bos_token

        if model_dir is not None:
            self.model_dir = model_dir
        if self.model_dir is None:
            raise ValueError("model_path is not set")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading model...")

        self.model = torch.load(
            path.join(self.model_dir, "consolidated.00.pth"),
            map_location=device,
        )

        print("Model loaded")

        with open(path.join(self.model_dir, "params.json"), "r") as f:
            self.config = json.load(f)

        self.config["rope_theta"] = torch.tensor(
            self.config["rope_theta"]
        )  # For convenience in RoPE calculations

        # Note: head_dim is standard size per head, which is 128 in llama3
        # head_dim is only used for scaling in attention calculation
        self.config["head_dim"] = 128

    def get_token_embeddings(self, prompt_tokens: list[int]) -> torch.Tensor:
        assert self.model is not None
        embedding_layer = torch.nn.Embedding(
            self.config["vocab_size"], self.config["dim"]
        )
        embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])
        tokens = torch.tensor(prompt_tokens)
        token_embeddings = embedding_layer(tokens).to(torch.bfloat16)
        print(token_embeddings.shape)
        return token_embeddings

    def scaled_dot_product_attn(
        self,
        q_layer_head: torch.Tensor,
        k_layer_head: torch.Tensor,
        v_layer_head: torch.Tensor,
        layer_embedding_norm: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # Project embeddings to get query, key, and value vectors
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

        # Split into pairs for RoPE
        q_split = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        k_split = k_per_token.float().view(k_per_token.shape[0], -1, 2)

        # Convert to complex numbers
        q_complex = torch.view_as_complex(q_split)
        k_complex = torch.view_as_complex(k_split)

        # Apply RoPE rotation
        q_rotated_complex = q_complex * freqs_cis
        k_rotated_complex = k_complex * freqs_cis

        # Convert back to real
        q_rotated = torch.view_as_real(q_rotated_complex).view(q_per_token.shape)
        k_rotated = torch.view_as_real(k_rotated_complex).view(k_per_token.shape)

        # Compute attention scores
        qk_per_token = torch.matmul(q_rotated, k_rotated.T) / (
            self.config["head_dim"] ** 0.5
        )

        # Create and apply attention mask
        mask = torch.full(
            (layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]),
            float("-inf"),
            device=layer_embedding_norm.device,
        )
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask

        # Apply softmax
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(
            qk_per_token_after_masking, dim=1
        ).to(torch.bfloat16)

        # Compute final attention output
        qkv_attention = torch.matmul(
            qk_per_token_after_masking_after_softmax, v_per_token
        )

        return qkv_attention

    def propagate_layer(self, layer_idx: int, embeddings: torch.Tensor):
        assert self.model is not None

        n_heads = self.config["n_heads"]
        n_kv_heads = self.config["n_kv_heads"]

        # Normalize for attention
        layer_embedding_norm = utils.rms_norm(
            embeddings,
            self.model[f"layers.{layer_idx}.attention_norm.weight"],
            self.config["norm_eps"],
        )

        # Get and reshape weight matrices
        q_layer = self.model[f"layers.{layer_idx}.attention.wq.weight"]
        q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, self.config["dim"])

        k_layer = self.model[f"layers.{layer_idx}.attention.wk.weight"]
        k_layer = k_layer.view(
            n_kv_heads, k_layer.shape[0] // n_kv_heads, self.config["dim"]
        )

        v_layer = self.model[f"layers.{layer_idx}.attention.wv.weight"]
        v_layer = v_layer.view(
            n_kv_heads, v_layer.shape[0] // n_kv_heads, self.config["dim"]
        )

        # Create frequencies for RoPE - calculate per pair position
        zero_to_one = torch.tensor(range(64)) / 64  # 64 pairs for 128-dim vector
        freqs = 1.0 / (self.config["rope_theta"] ** zero_to_one)

        # Create position-specific rotation matrices
        freqs_for_each_token = torch.outer(
            torch.arange(embeddings.shape[0], device=embeddings.device), freqs
        )
        freqs_cis = torch.polar(
            torch.ones_like(freqs_for_each_token), freqs_for_each_token
        )

        qkv_attention_store = []

        # Process each attention head
        for head_idx in range(n_heads):
            q_layer_head = q_layer[head_idx]
            k_layer_head = k_layer[head_idx // 4]  # Share KV heads across 4 Q heads
            v_layer_head = v_layer[head_idx // 4]

            qkv_attention_store.append(
                self.scaled_dot_product_attn(
                    q_layer_head,
                    k_layer_head,
                    v_layer_head,
                    layer_embedding_norm,
                    freqs_cis,
                )
            )

        # Merge attention heads
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)

        # Project back to embedding dimension
        w_layer = self.model[f"layers.{layer_idx}.attention.wo.weight"]
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)

        # Add residual connection
        embedding_after_edit = embeddings + embedding_delta

        # Normalize for feed-forward
        embedding_after_edit_normalized = utils.rms_norm(
            embedding_after_edit,
            self.model[f"layers.{layer_idx}.ffn_norm.weight"],
            self.config["norm_eps"],
        )

        # Feed-forward network
        w1 = self.model[f"layers.{layer_idx}.feed_forward.w1.weight"]
        w2 = self.model[f"layers.{layer_idx}.feed_forward.w2.weight"]
        w3 = self.model[f"layers.{layer_idx}.feed_forward.w3.weight"]

        # SwiGLU activation
        output_after_feedforward = torch.matmul(
            torch.nn.functional.silu(
                torch.matmul(embedding_after_edit_normalized, w1.T)
            )
            * torch.matmul(embedding_after_edit_normalized, w3.T),
            w2.T,
        )

        # Final residual connection
        final_embedding = embedding_after_edit + output_after_feedforward

        return final_embedding

    def generate(self, tokens: list[int]) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded")

        # Get initial token embeddings (unnormalized)
        final_embedding = self.get_token_embeddings(tokens)

        # Process through all layers
        for layer in range(self.config["n_layers"]):
            final_embedding = self.propagate_layer(layer, final_embedding)

        # Final normalization
        final_embedding = utils.rms_norm(
            final_embedding, self.model["norm.weight"], self.config["norm_eps"]
        )

        return final_embedding
