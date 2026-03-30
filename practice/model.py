import torch
import json
from os import path
import utils


class LlamaModel:
    """
    Implementation of the Llama language model architecture.
    This class loads a pre-trained Llama model and provides methods for token embedding generation,
    attention mechanisms, and text generation.
    """

    def __init__(self, model_dir: str | None = None, bos_token: int = 128000):
        """
        Initialize the Llama model.

        Args:
            model_dir (str | None): Directory containing model weights and parameters.
            bos_token (int): The beginning-of-sequence token ID.

        Process:
            1. Sets model directory and BOS token
            2. Loads model weights from consolidated file
            3. Loads model configuration parameters from params.json
            4. Sets up RoPE and head dimensions

        Raises:
            ValueError: If model_path is not set
        """
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
        """
        Convert token IDs to their corresponding embeddings.

        Args:
            prompt_tokens (list[int]): List of token IDs to embed.

        Process:
            1. Creates an PyTorch embedding layer with model vocab size and dimension
            2. Copies weights from the model's token embedding weights
            3. Converts input tokens to tensor and applies embedding
            4. Converts output to bfloat16 precision

        Returns:
            torch.Tensor: Token embeddings with shape [seq_len, embedding_dim]
        """
        assert self.model is not None
        embedding_layer = torch.nn.Embedding(
            self.config["vocab_size"], self.config["dim"]
        )
        embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])
        # TODO: Create tensor from token_ids and apply embedding
        pass

    def scaled_dot_product_attn(
        self,
        q_layer_head: torch.Tensor,
        k_layer_head: torch.Tensor,
        v_layer_head: torch.Tensor,
        layer_embedding_norm: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Compute scaled dot-product attention with rotary positional embeddings (RoPE).

        Args:
            q_layer_head (torch.Tensor): Query weight matrix for current attention head
            k_layer_head (torch.Tensor): Key weight matrix for current attention head
            v_layer_head (torch.Tensor): Value weight matrix for current attention head
            layer_embedding_norm (torch.Tensor): Normalized input embeddings
            freqs_cis (torch.Tensor): Complex rotation frequencies for RoPE

        Process:
            1. Project embeddings to get query, key, and value vectors
            2. Split vectors into pairs for RoPE and convert to complex numbers
            3. Apply RoPE rotation to queries and keys
            4. Convert back to real numbers
            5. Compute attention scores and scale by head dimension
            6. Apply causal mask to prevent attending to future tokens
            7. Apply softmax to get attention weights
            8. Compute final attention output by weighted sum of values

        Returns:
            torch.Tensor: Attention output for the current head
        """
        # Project embeddings to get query, key, and value vectors
        q_per_token = None
        k_per_token = None
        v_per_token = None

        # Rotate via RoPE
        q_rotated = utils.rope_rotate(q_per_token, freqs_cis)
        k_rotated = utils.rope_rotate(k_per_token, freqs_cis)

        # Compute attention scores
        qk_per_token = torch.matmul(q_rotated, k_rotated.T) / (
            self.config["head_dim"] ** 0.5
        )

        # Create and apply attention mask
        mask = None
        qk_per_token_after_masking = qk_per_token + mask

        # Apply softmax
        qk_per_token_after_masking_after_softmax = None

        # Compute final attention output
        qkv_attention = torch.matmul(
            qk_per_token_after_masking_after_softmax, v_per_token
        )

        return qkv_attention

    def propagate_layer(self, layer_idx: int, embeddings: torch.Tensor):
        """
        Process embeddings through a single transformer layer.

        Args:
            layer_idx (int): Index of the layer to process
            embeddings (torch.Tensor): Token embeddings to process

        Process:
            1. Normalize embeddings for attention using RMS normalization
            2. Extract and reshape QKV weight matrices
            3. Generate RoPE frequencies for position encoding
            4. Process each attention head, applying scaled dot-product attention
            5. Concatenate attention heads and project back to embedding dimension
            6. Add residual connection
            7. Normalize for feed-forward network
            8. Apply feed-forward network with SwiGLU activation
            9. Add final residual connection

        Returns:
            torch.Tensor: Processed embeddings after this layer
        """
        assert self.model is not None

        n_heads = self.config["n_heads"]
        n_kv_heads = self.config["n_kv_heads"]

        # Normalize for attention
        layer_embedding_norm = None

        # Get and reshape weight matrices
        q_layer = self.model[f"layers.{layer_idx}.attention.wq.weight"]
        q_layer = q_layer.view("TODO: Figure out what shape to use")

        k_layer = self.model[f"layers.{layer_idx}.attention.wk.weight"]
        k_layer = k_layer.view("TODO: Figure out what shape to use")

        v_layer = self.model[f"layers.{layer_idx}.attention.wv.weight"]
        v_layer = v_layer.view("TODO: Figure out what shape to use")

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
            q_layer_head = None
            k_layer_head = None
            v_layer_head = None

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
        stacked_qkv_attention = None

        # Project back to embedding dimension
        w_layer = None
        embedding_delta = None

        # Add residual connection
        embedding_after_edit = None

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
        output_after_feedforward = None

        # Final residual connection
        final_embedding = embedding_after_edit + output_after_feedforward

        return final_embedding

    def generate(self, tokens: list[int]) -> torch.Tensor:
        """
        Process tokens through the entire model to get final embeddings.

        Args:
            tokens (list[int]): List of token IDs to process

        Process:
            1. Check if model is loaded
            2. Get initial token embeddings
            3. Process embeddings through each model layer sequentially
            4. Apply final normalization

        Returns:
            torch.Tensor: Final processed embeddings after all layers

        Raises:
            ValueError: If model is not loaded
        """
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
