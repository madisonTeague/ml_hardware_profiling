"""
Standalone NCU driver script for profiling pruned model forward passes.

Invoked by modal_pruning.py under NSight Compute.  Not meant to be run
directly -- use ``modal run modal_pruning.py --command ncu`` instead.

Accepts configuration via environment variables so that modal_pruning.py
does not need to template Python source as a string:

    MODEL_PATH   -- HuggingFace name or local path   (required)
    BATCH_SIZE   -- batch size                        (default: 1)
    SEQ_LEN      -- sequence length                   (default: 2048)
"""

import os
import sys

sys.path.insert(0, "/app")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.environ["MODEL_PATH"]
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
seq_len = int(os.environ.get("SEQ_LEN", "2048"))

torch.cuda.cudart().cudaProfilerStart()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.float16, device_map="auto",
    rope_scaling={"type": "dynamic", "factor": 3.0}
)
model.eval()

input_ids = torch.randint(
    0, tokenizer.vocab_size,
    (batch_size, seq_len), device="cuda",
)

# Warm-up (outside profiled region via cudaProfiler markers)
with torch.no_grad():
    model(input_ids)
torch.cuda.synchronize()

# Profiled forward pass
with torch.no_grad():
    model(input_ids)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStop()
