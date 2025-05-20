import torch
from tqdm import tqdm
def gpt_generate(model, tokenizer, prompt, max_length=50, temperature=0.7) -> str:
    """
    Generate text from a prompt using the model
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)
    for _ in tqdm(range(max_length)):
        output = model(tokens).logits
        if isinstance(output, tuple):
            output = output[0]
        logits = output[:, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokenizer.decode_sequence(tokens[0].tolist())

def generate(model, tokenizer, prompt, max_length=50, temperature=0.7) -> str:
    """
    Generate text from a prompt using the model
    """
    return generate_with_saes(model, tokenizer, prompt, max_length=max_length, temperature=temperature, activations_to_patch=[])

def generate_with_saes(model, tokenizer, prompt, max_length=50, temperature=0.7, activations_to_patch: list[str] = []) -> str:
    """
    Generate text from a prompt using the model
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt)
    tokens = torch.Tensor(tokens).long().unsqueeze(0).to(device)
    for _ in range(max_length):
        with model.use_saes(activations_to_patch=activations_to_patch):
            output = model(tokens).logits
        if isinstance(output, tuple):
            output = output[0]
        logits = output[:, -1]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokenizer.decode_sequence(tokens[0].tolist())