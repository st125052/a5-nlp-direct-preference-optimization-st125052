import torch
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    model_name_or_path = "st125052/a5-dpo"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

def get_prediction(prompt):
    device = get_torch_device()

    model, tokenizer = get_model_and_tokenizer()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    return list(tokenizer.decode(outputs[0], skip_special_tokens=True))