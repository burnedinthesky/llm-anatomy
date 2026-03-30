import torch
from model import LlamaModel
import utils


def main():
    tokenizer = utils.init_tokenizer("../model/tokenizer.model")
    model = LlamaModel("../model")
    assert model.model is not None

    prompt = (
        "the answer to the ultimate question of life, the universe, and everything is "
    )

    tokens = tokenizer.encode(prompt)
    tokens = [model.bos_token] + tokens

    print(f"Tokens: {tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([token]) for token in tokens]}")

    embeddings = model.generate(tokens)
    last_token_embedding = embeddings[-1]
    logits = torch.matmul(last_token_embedding, model.model["output.weight"].T)
    logits = logits / 1.0  # You can adjust temperature here

    next_token = torch.argmax(logits)
    next_token_id = int(next_token.item())
    next_token_text = tokenizer.decode([next_token_id])

    print(f"\nPrompt: {prompt}")
    print(f"Model's continuation: {next_token_text}")
    print(f"Next token ID: {next_token_id}")

    top_k = 5
    top_logits, top_indices = torch.topk(logits, top_k)
    print("\nTop 5 predictions:")
    for i in range(top_k):
        token = tokenizer.decode([int(top_indices[i].item())])
        print(f"{i+1}. {token} (logit: {top_logits[i].item():.2f})")


if __name__ == "__main__":
    main()
