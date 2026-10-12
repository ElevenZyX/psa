"""
Prompt-format and read-token diagnostic (tokenizer only, no model is loaded).

Builds the (positive, negative) pair for one example of a PSA dataset and one example of
the TOFU dataset with `make_pos_neg_pair`, prints both prompts with repr() for visual
inspection, and checks which token falls at `read_token_index` (run.py uses -2).

    python check_prompt_format.py
    python check_prompt_format.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --psa-dataset sycophancy --own-dataset tofu-forget10 --index 0
    python check_prompt_format.py --selftest   # chat date: 1B (Llama-3.2) and 8B (Llama-3.1) tokenizers only, CPU

--selftest checks, on the first SELFTEST_N examples of the 8 datasets and both answers:
  (1) the prompts built by make_pos_neg_pair with the 1B and the 8B tokenizers contain "Today Date: 26 Jul 2024"
      (utils.CHAT_DATE) exactly once;
  (2) with the 8B tokenizer the prompt with date_string=CHAT_DATE is identical to the prompt without date_string
      (the template default), so the 8B caches extracted before CHAT_DATE existed are still valid.
  For the 1B it also prints what the template writes without date_string (the day it runs: the bug being fixed).
"""
import argparse
import datetime
import json
import re

from transformers import AutoTokenizer

from cache_alignment import DEFAULT_DATASETS
from utils import (CHAT_DATE, TODAY_DATE_PREFIX, _format_llama3, check_today_date, make_pos_neg_pair,
                   diagnose_read_token_index, is_llama3_tokenizer)

SELFTEST_MODELS = {   # model -> the template's default date is already CHAT_DATE (Llama-3.1) or strftime_now (Llama-3.2)
    'open-unlearning/tofu_Llama-3.2-1B-Instruct_full': False,
    'open-unlearning/tofu_Llama-3.1-8B-Instruct_full': True,
}
SELFTEST_N = 5


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
              f"(fixed to utils.CHAT_DATE={CHAT_DATE!r}; identical in pos and neg, but it is context for both, "
              f"so it does NOT cancel in pos - neg)")
    ok_pos = diagnose_read_token_index(pos, tokenizer, read_token_index,
                                       expected_letter=mwe['answer_matching_behavior'][1], label='[pos] ')
    ok_neg = diagnose_read_token_index(neg, tokenizer, read_token_index,
                                       expected_letter=mwe['answer_not_matching_behavior'][1], label='[neg] ')
    return ok_pos and ok_neg


def selftest() -> None:
    expected = TODAY_DATE_PREFIX + CHAT_DATE
    today = TODAY_DATE_PREFIX + datetime.datetime.now().strftime("%d %b %Y")   # what strftime_now("%d %b %Y") gives
    for model, default_is_chat_date in SELFTEST_MODELS.items():
        print("=" * 100)
        tok = AutoTokenizer.from_pretrained(model)
        assert is_llama3_tokenizer(tok), f"{model}: not routed to the Llama-3 branch of make_pos_neg_pair"
        n_checked, n_identical, defaults = 0, 0, set()
        for ds in DEFAULT_DATASETS:
            data = json.load(open(f"./datasets/generate/{ds}/generate_dataset.json", 'r', encoding='utf-8'))
            for mwe in data[:SELFTEST_N]:
                pos, neg = make_pos_neg_pair(mwe, tok)
                for p, ans in ((pos, mwe['answer_matching_behavior']), (neg, mwe['answer_not_matching_behavior'])):
                    lines = [l for l in p.split('\n') if l.startswith(TODAY_DATE_PREFIX)]
                    assert lines == [expected], f"{model} {ds}: Today Date lines {lines!r}, expected [{expected!r}]"
                    assert p == _format_llama3(tok, mwe['question'], ans, date_string=CHAT_DATE)
                    default = _format_llama3(tok, mwe['question'], ans, date_string=None)
                    defaults.update(l for l in default.split('\n') if l.startswith(TODAY_DATE_PREFIX))
                    n_identical += int(default == p)
                    n_checked += 1
        print(f"{model}: {n_checked} prompts ({SELFTEST_N} examples x 8 datasets x 2 answers) contain {expected!r} exactly once")
        print(f"  without date_string the template writes {sorted(defaults)!r}; identical to the CHAT_DATE prompt in "
              f"{n_identical}/{n_checked}")
        if default_is_chat_date:
            assert n_identical == n_checked, f"{model}: date_string=CHAT_DATE changes {n_checked - n_identical} prompts"
            print(f"  OK: date_string=CHAT_DATE does not change any {model.split('/')[-1]} prompt -> its caches stay valid")
        else:
            print(f"  (today's strftime_now would give {today!r}: "
                  + ("matches the template default -> this template dates the prompt with the day it runs)"
                     if sorted(defaults) == [today] else "does NOT match; check the template by hand)"))
        pos, neg = make_pos_neg_pair(json.load(open("./datasets/generate/tofu-forget10/generate_dataset.json",
                                                    'r', encoding='utf-8'))[0], tok)
        check_today_date(pos, '[pos] ')
        assert diagnose_read_token_index(pos, tok, -2, label='[pos] ')
    # the diagnostic must stop on a prompt dated by the run day
    try:
        check_today_date(pos.replace(expected, TODAY_DATE_PREFIX + "12 Oct 2026"), '[corrupted] ')
    except SystemExit as e:
        print(f"(expected) a prompt with another date stops the diagnostic: {str(e)[:80]}...")
    else:
        raise AssertionError("check_today_date accepted a wrong date")
    print("=" * 100)
    print("SELFTEST PASSED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    parser.add_argument("--psa-dataset", default="sycophancy")
    parser.add_argument("--own-dataset", default="tofu-forget10")
    parser.add_argument("--index", type=int, default=0, help="example index within each dataset")
    parser.add_argument("--read-token-index", type=int, default=-2, help="same value run.py uses")
    parser.add_argument("--selftest", action='store_true', help="chat-date checks with the 1B and 8B tokenizers (CPU)")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        raise SystemExit(0)

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
