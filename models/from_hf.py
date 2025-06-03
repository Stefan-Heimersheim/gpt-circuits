"""
python -m models.from_hf --model_name gpt2
"""

from config.gpt.models import gpt_options
from models.gpt import GPT

gpt2 = GPT.from_pretrained("gpt2")
gpt2.save("checkpoints/gpt2")
