"""
Prompt-format and read-token diagnostic (tokenizer only, no model is loaded).

Builds the (positive, negative) pair for one example of a PSA dataset and one example of
the TOFU dataset with `make_pos_neg_pair`, prints both prompts with repr() for visual
inspection, and checks which token falls at `read_token_index` (run.py uses -2).

    python check_prompt_format.py
    python check_prompt_format.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --psa-dataset sycophancy --own-dataset tofu-forget10 --index 0
"""
import argparse
import json
import re

from transformers import AutoTokenizer

from utils import make_pos_neg_pair, diagnose_read_token_index, is_llama3_tokenizer


def load_example(dataset: str, index: int) -> dict:
    data = json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", 'r', encoding='utf-8'))
    return data[index]


def check(dataset: str, index: int, tokenizer, read_token_index: int) -> bool:
    mwe = load_example(dataset, index)
    pos, neg = make_pos_neg_pair(mwe, tokenizer)
    print("=" * 100)
    print(f"dataset={dataset!r} index={index} answer_matching={mwe['answer_matching_behavior']!r} "
          f"answer_not_matching={mwe['answer_not_matching_behavior']!r}")
    print("-" * 100)
    print("POS prompt:")
    print(repr(pos))
    print("-" * 100)
    print("NEG prompt:")
    print(repr(neg))
    print("-" * 100)
    date_line = re.search(r"Today Date: [^\n]*", pos)
    if date_line:
        print(f"NOTE: chat template inserted a date block -> {date_line.group(0)!r} "
              f"(identical in pos and neg, cancels in pos - neg)")
    ok_pos = diagnose_read_token_index(pos, tokenizer, read_token_index,
                                       expected_letter=mwe['answer_matching_behavior'][1], label='[pos] ')
    ok_neg = diagnose_read_token_index(neg, tokenizer, read_token_index,
                                       expected_letter=mwe['answer_not_matching_behavior'][1], label='[neg] ')
    return ok_pos and ok_neg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    parser.add_argument("--psa-dataset", default="sycophancy")
    parser.add_argument("--own-dataset", default="tofu-forget10")
    parser.add_argument("--index", type=int, default=0, help="example index within each dataset")
    parser.add_argument("--read-token-index", type=int, default=-2, help="same value run.py uses")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"tokenizer.name_or_path = {tokenizer.name_or_path!r} | is_llama3 = {is_llama3_tokenizer(tokenizer)}")
    print(f"bos={tokenizer.bos_token!r} ({tokenizer.bos_token_id}) eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id})")

    ok = True
    for dataset in (args.psa_dataset, args.own_dataset):
        ok = check(dataset, args.index, tokenizer, args.read_token_index) and ok

    print("=" * 100)
    if ok:
        print(f"ALL OK: read_token_index={args.read_token_index} lands on the answer letter for both datasets.")
    else:
        print(f"PROBLEM: read_token_index={args.read_token_index} does NOT land on the answer letter. "
              f"Adjust it before extracting vectors.")
        raise SystemExit(1)
