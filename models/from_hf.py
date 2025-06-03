"""
python -m models.from_hf --model_name gpt2
"""

from config.gpt.models import gpt_options
from models.gpt import GPT

gpt = GPT.from_pretrained("gpt2")
gpt.save("checkpoints_gpt2/gpt2")

