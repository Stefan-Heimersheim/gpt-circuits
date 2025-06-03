"""
python -m models.from_hf --model_name gpt2
"""

import argparse
from config.gpt.models import gpt_options
from models.gpt import GPT

def main():
    parser = argparse.ArgumentParser(description="Load model from HuggingFace and save locally")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name to load from HuggingFace")
    args = parser.parse_args()
    
    model = GPT.from_pretrained(args.model_name)
    model.save(f"checkpoints/{args.model_name}")

if __name__ == "__main__":
    main()
