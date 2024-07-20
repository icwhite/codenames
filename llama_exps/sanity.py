import fire
import time
import argparse
import pandas as pd
import numpy as np
import ast

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_exps.utils import add_player_demo_col, get_fixed_generation_prob, load_model_and_tokenizer, ALL_DEMOS

model, tokenizer = load_model_and_tokenizer('llama2-7b')

res = get_fixed_generation_prob(model,
                              tokenizer,
                              sequences=["What is something in space?", "What is a small animal?"],
                              responses=["astronaut", "ant"])

print(res)