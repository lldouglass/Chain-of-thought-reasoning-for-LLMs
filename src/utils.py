from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(name: str = 'gpt2', dtype: str = 'float16') -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    _dtype = torch.float16 if dtype == 'float16' and DEVICE == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=_dtype,
        device_map='auto' if DEVICE == 'cuda' else None,
        output_hidden_states=True,
    ).eval()
    return model, tok


def load_prompts(path: str | Path) -> List[str]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'prompts' in data:
        return data['prompts']  # type: ignore[index]
    raise ValueError("JSON must be list or contain 'prompts'")

print('Utils loaded — functions are now available in the namespace.')