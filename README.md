# LLM Anatomy: Implement Llama 3.1 8B from Scratch

A hands-on workshop where you implement the Llama 3.1 8B architecture from scratch in PyTorch, building every component — token embeddings, RoPE, multi-head attention with GQA, SwiGLU feed-forward networks, and RMS normalization — to understand how a modern LLM works under the hood.

Based on [llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) by [@naklecha](https://github.com/naklecha). Adapted for the NYCU SDC workshop.

## Prerequisites

- Python 3.10+
- PyTorch
- `tiktoken`
- Llama 3.1 8B model weights (`consolidated.00.pth`, `params.json`, `tokenizer.model`) placed in a `model/` directory

Install dependencies:

```bash
pip install torch tiktoken
```

## Project Structure

```
.
├── main.py          # Entry point — tokenizes a prompt, runs the model, prints predictions
├── model.py         # Complete Llama model implementation
├── utils.py         # Utilities: RMS norm, RoPE rotation, tokenizer init
├── practice/        # Workshop version with blanks to fill in
│   ├── main.py
│   ├── model.py     # Fill-in-the-blank model (TODOs and empty fields)
│   ├── utils.py     # Fill-in-the-blank utilities
│   └── notebooks/
│       └── 2_understand_llama3.ipynb  # Notebook for exploring the model file
└── model/           # (not tracked) Place Llama 3.1 8B weights here
```

## Usage

Run the complete implementation:

```bash
python main.py
```

## Workshop

Work through the practice version in `practice/` — the code has TODO comments and empty assignments that guide you through implementing each component:

1. **Token embeddings** — convert token IDs to vectors
2. **RMS normalization** — pre-norm used throughout the transformer
3. **Rotary positional embeddings (RoPE)** — encode position via complex rotation
4. **Scaled dot-product attention** — QKV projection, masking, softmax
5. **Grouped-query attention (GQA)** — KV head sharing across Q heads
6. **SwiGLU feed-forward network** — gated activation in the FFN block
7. **Residual connections** — skip connections between sub-layers

Run from the `practice/` directory:

```bash
cd practice
python main.py
```

The notebook in `practice/notebooks/` helps you explore the model file structure before implementing.

## Acknowledgments

This project is based on [llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) by [Nishant Aklecha](https://github.com/naklecha), which provides an excellent walkthrough of implementing Llama 3 from scratch. The original work is a detailed, educational implementation that inspired this workshop format.
