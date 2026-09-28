"""Load the untouched Prism sampler/verifier modules by file path (the dllm_eval package
pulls in sacrebleu etc., which we do not need), plus model, tokenizer and GSM8K prompts."""
import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRISM_DIR = ROOT / "LLaDA" / "LLaDA_Prism"
MODELS_DIR = PRISM_DIR / "dllm_eval" / "models"
TASKS_DIR = PRISM_DIR / "dllm_eval" / "tasks"

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
EOS_ID = 126081

# exactly scripts/run_gsm8k.sh -> LLaDA.generate_until (use_hts branch)
OFFICIAL_GSM8K_GEN_KWARGS = dict(
    initial_N=16, final_K=4, hts_survivor_k=2, hts_mode=True,
    hts_start_pct=0.1, hts_end_pct=0.6, decay_factor=1.8, pruning_interval=3,
    reward_mode="svf", task_type="math", steps=32, gen_length=256, block_length=32,
    temperature=0.7, top_p=0.95, top_k=None, threshold=0.85,
    mask_id=MASK_ID, eos_id=EOS_ID, until=["[NO_UNTIL_PLACEHOLDER]"],
)


def _load_module(fullname, path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_prism_modules():
    if "prism_models" not in sys.modules:
        pkg = types.ModuleType("prism_models")
        pkg.__path__ = [str(MODELS_DIR)]
        sys.modules["prism_models"] = pkg
        _load_module("prism_models.verifier", MODELS_DIR / "verifier.py")
        _load_module("prism_models.hts_sampler", MODELS_DIR / "hts_sampler.py")
    return sys.modules["prism_models.hts_sampler"], sys.modules["prism_models.verifier"]


def gsm_prompt(doc):
    mod = _load_module("prism_gsm8k_utils", TASKS_DIR / "gsm8k" / "utils.py")
    return mod.gsm_prompt(doc)


def load_gsm8k_test():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="test")


def load_model_and_tokenizer(device="mps"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device).to(torch.bfloat16).eval()
    return model, tok


def encode_prompt(tok, prompt_text, device):
    # LLaDA.tok_batch_encode: add_special_tokens=False (add_bos_token=False), padding longest
    enc = tok([prompt_text], truncation=False, padding="longest", return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def set_seed(seed):
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
