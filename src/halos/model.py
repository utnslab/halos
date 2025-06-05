from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
import torch

def load_pythia_model(model_name):
    if is_flash_attn_2_available():
        model = AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            revision="step0",
            attn_implementation="flash_attention_2",
            use_cache=False)
        return model
    else:
        return AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            revision="step0",
            use_cache=False)
