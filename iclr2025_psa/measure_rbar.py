"""
Measure R̄ (rbar) per layer for several contrastive datasets and one model.

For each dataset and each layer:
    diff_i     = h_pos_i - h_neg_i                    (residual stream at read_token_index)
    norm_mean  = || mean_i diff_i ||
    mean_norm  = mean_i || diff_i ||
    rbar       = norm_mean / mean_norm                (in [0, 1]; 1 = all diffs perfectly aligned)
    n          = number of pairs

This is the clean measurement (no clipping, no noise), i.e. what the DP mechanism in run.py
would see before privatising. run.py is not touched; the activations are extracted with the
same call run.py uses (steering_vectors.train_steering_vector, read_token_index=-2).

    python measure_rbar.py
    python measure_rbar.py --model open-unlearning/tofu_Llama-3.2-3B-Instruct_full --out-dir rbar_3b
    python measure_rbar.py --datasets sycophancy tofu-forget10 --layers 4 8 12 --n-samples 100

Outputs (in --out-dir): rbar_results.csv, rbar_by_layer.png, norm_mean_by_layer.png
"""
import argparse
import json
import random
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from steering_vectors import (
    train_steering_vector,
    get_num_matching_layers,
    guess_and_enhance_layer_config,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import make_dataset, make_pos_neg_pair, diagnose_read_token_index, device

PSA_DATASETS = [
    'sycophancy', 'hallucination', 'refusal', 'myopic-reward',
    'survival-instinct', 'coordinate-other-ais', 'corrigible-neutral-HHH',
]
OWN_DATASET = 'tofu-forget10'
DEFAULT_DATASETS = PSA_DATASETS + [OWN_DATASET]
READ_TOKEN_INDEX = -2  # same as run.py
CSV_COLUMNS = ['model', 'dataset', 'layer', 'rbar', 'norm_mean', 'mean_norm', 'n']

# Categorical palette (fixed slot order) for the 7 alignment datasets; tofu-forget10 is drawn
# in black, thicker and with markers so it stands out against the seven.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
SERIES_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


# --------------------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------------------
class RbarAggregator:
    """
    Aggregator for steering_vectors.train_steering_vector. Returns the plain mean of the
    difference vectors (no clipping, no noise) and records rbar / norm_mean / mean_norm / n for
    every call.

    Layer <-> call correspondence. The aggregator only receives (pos, neg) and is not told the
    layer. Two facts from steering_vectors==0.12.1 are used:

      (1) train_steering_vector returns SteeringVector.layer_activations: dict[int, Tensor],
          and the Tensor stored for layer L is *the very object* this aggregator returned for
          that layer (train_steering_vector.py: aggregate_activations, line 152-155 stores
          `direction_vec` as returned; line 209 wraps the dict in SteeringVector untouched).
          So each record keeps a reference to the returned tensor and, after training, layer L
          is matched to its record by object identity (`is`). This is the mapping that is
          actually used: it is unambiguous and does not depend on call order at all.

      (2) As a cross-check only, the observed call order is compared with the expected one.
          The library calls the aggregator in ASCENDING layer index, regardless of the order
          of the `layers` argument: record_activations.py:52-59 registers one forward hook
          per selected layer and the defaultdict `recorded_activations` gets its keys when the
          hooks fire (record_activations.py:80), i.e. in forward order of the decoder blocks;
          train_steering_vector.py:246 (`for layer_num, activation in record.items()`),
          :117-124 (`pos_acts_by_layer[layer_num].append`) and :149
          (`for layer_num in pos_acts_by_layer.keys()`) all preserve that insertion order.
    """

    def __init__(self):
        self.records: list[tuple[torch.Tensor, dict]] = []  # (returned mean_vec, stats), in call order

    def __call__(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        assert pos.dtype == torch.float32 and neg.dtype == torch.float32, \
            f"activations must be float32 for a clean norm ratio, got {pos.dtype}/{neg.dtype}"
        diff = pos - neg                       # (n, hidden)
        mean_vec = torch.mean(diff, dim=0)     # (hidden,)
        norms = torch.norm(diff, dim=1)        # (n,)
        norm_mean = torch.norm(mean_vec).item()
        mean_norm = norms.mean().item()
        stats = {
            'rbar': norm_mean / mean_norm,
            'norm_mean': norm_mean,
            'mean_norm': mean_norm,
            'n': int(diff.shape[0]),
        }
        self.records.append((mean_vec, stats))
        return mean_vec

    def stats_for(self, vec: torch.Tensor) -> dict:
        matches = [stats for (t, stats) in self.records if t is vec]
        assert len(matches) == 1, f"expected exactly one record for this layer vector, found {len(matches)}"
        return matches[0]

    def call_order(self, layer_activations: dict[int, torch.Tensor]) -> list[int]:
        """Layer index of each aggregator call, in the order the calls happened."""
        vec_to_layer = {id(v): layer for layer, v in layer_activations.items()}
        return [vec_to_layer[id(t)] for (t, _) in self.records]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def resolve_dataset_path(name: str) -> Path:
    """`name` is either a dataset under ./datasets/generate/ or a path to an A/B json file."""
    p = Path(name)
    if p.suffix == '.json' and p.is_file():
        return p
    return Path('datasets') / 'generate' / name / 'generate_dataset.json'


def load_pairs(name: str, n_samples: int, seed: int) -> tuple[list[dict], int]:
    """Shuffle with `seed` (same shuffle run.py does with random.seed(42)) and keep n_samples."""
    path = resolve_dataset_path(name)
    data = json.load(open(path, 'r', encoding='utf-8'))
    total = len(data)
    random.Random(seed).shuffle(data)
    if total < n_samples:
        print(f"WARNING: dataset {name!r} has only {total} pairs (< n_samples={n_samples}); "
              f"using all {total}. Its rbar is NOT directly comparable with the {n_samples}-pair datasets.")
        return data, total
    return data[:n_samples], total


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------
def measure_dataset(name: str, model, tokenizer, layers: list[int], n_samples: int, seed: int,
                    model_name: str) -> list[dict]:
    mwe_data, total = load_pairs(name, n_samples, seed)
    pairs = make_dataset(mwe_data, tokenizer)
    print(f"[{name}] {len(pairs)} pairs (dataset size {total}), layers {layers[0]}..{layers[-1]} "
          f"({len(layers)} layers)")

    aggregator = RbarAggregator()
    sv = train_steering_vector(
        model,
        tokenizer,
        pairs,
        read_token_index=READ_TOKEN_INDEX,
        show_progress=True,
        aggregator=aggregator,
        layers=layers,
        tqdm_desc=f"Extracting {name}",
    )

    got = sorted(sv.layer_activations.keys())
    assert got == sorted(layers), f"layers returned by the library {got} != requested {sorted(layers)}"
    assert len(aggregator.records) == len(layers), \
        f"aggregator was called {len(aggregator.records)} times for {len(layers)} layers"

    order = aggregator.call_order(sv.layer_activations)
    if order != sorted(layers):
        # Not fatal: the identity mapping below is what is used. It only means fact (2) in the
        # RbarAggregator docstring no longer holds for this library version.
        print(f"WARNING: aggregator call order {order} is not ascending layer order; "
              f"results still use the identity mapping and are correct.")

    rows = []
    for layer in sorted(layers):
        stats = aggregator.stats_for(sv.layer_activations[layer])
        rows.append({'model': model_name, 'dataset': name, 'layer': layer, **stats})
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)


def print_table(rows: list[dict], dataset_order: list[str]) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    df = df.sort_values(['dataset', 'layer'])
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', '{:.6f}'.format):
        print(df.to_string(index=False))


def plot_metric(rows: list[dict], dataset_order: list[str], metric: str, ylabel: str, title: str,
                path: Path) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    alignment = [d for d in dataset_order if d != OWN_DATASET and d in set(df['dataset'])]
    for i, name in enumerate(alignment):
        sub = df[df['dataset'] == name].sort_values('layer')
        ax.plot(sub['layer'], sub[metric], color=SERIES_COLORS[i % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[i % len(SERIES_MARKERS)], markersize=4, linewidth=1.5,
                label=name)
    if OWN_DATASET in set(df['dataset']):
        sub = df[df['dataset'] == OWN_DATASET].sort_values('layer')
        ax.plot(sub['layer'], sub[metric], color='black', marker='o', markersize=6, linewidth=3.2,
                label=OWN_DATASET, zorder=10)
    ax.set_xlabel('layer index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df['layer'].unique()))
    if metric == 'rbar':
        ax.set_ylim(0, 1)
    ax.grid(True, color='#e5e4e0', linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    parser.add_argument("--datasets", nargs='+', default=DEFAULT_DATASETS,
                        help="dataset names under ./datasets/generate/ or paths to A/B json files")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help="layer indices; default = all layers (model.config.num_hidden_layers)")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".", help="where rbar_results.csv and the PNGs go")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'rbar_results.csv'

    # ---- model -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(device).eval()
    param_dtype = next(model.parameters()).dtype
    assert param_dtype == torch.float32, f"model loaded as {param_dtype}, expected float32"

    num_layers = model.config.num_hidden_layers
    layer_config = guess_and_enhance_layer_config(model, None, 'decoder_block')
    lib_num_layers = get_num_matching_layers(model, layer_config['decoder_block'])
    assert lib_num_layers == num_layers, \
        f"steering_vectors matched {lib_num_layers} decoder blocks ({layer_config['decoder_block']!r}) " \
        f"but config says {num_layers}"
    layers = args.layers if args.layers is not None else list(range(num_layers))
    assert all(0 <= l < num_layers for l in layers), f"layers {layers} out of range [0, {num_layers})"

    print("=" * 100)
    print(f"model          : {args.model}")
    print(f"num layers     : {num_layers} (steering_vectors decoder-block matcher: {layer_config['decoder_block']!r})")
    print(f"layers measured: {layers}")
    print(f"dtype          : {param_dtype}")
    print(f"device         : {next(model.parameters()).device}")
    print(f"read token idx : {READ_TOKEN_INDEX}")
    print(f"n_samples      : {args.n_samples} | seed: {args.seed}")
    print(f"datasets       : {args.datasets}")
    print(f"out dir        : {out_dir.resolve()}")
    print("=" * 100)

    # ---- example prompt, so the format can be eyeballed before the long run ----------
    first = json.load(open(resolve_dataset_path(args.datasets[0]), 'r', encoding='utf-8'))[0]
    pos, neg = make_pos_neg_pair(first, tokenizer)
    print(f"example prompt ({args.datasets[0]}, index 0)")
    print("POS:", repr(pos))
    print("NEG:", repr(neg))
    ok = diagnose_read_token_index(pos, tokenizer, READ_TOKEN_INDEX,
                                   expected_letter=first['answer_matching_behavior'][1], label='[pos] ')
    ok = diagnose_read_token_index(neg, tokenizer, READ_TOKEN_INDEX,
                                   expected_letter=first['answer_not_matching_behavior'][1], label='[neg] ') and ok
    if not ok:
        raise SystemExit(f"read_token_index={READ_TOKEN_INDEX} does not land on the answer letter; "
                         f"fix READ_TOKEN_INDEX before measuring.")
    print("=" * 100)

    # ---- per dataset -----------------------------------------------------------------
    all_rows: list[dict] = []
    errors: dict[str, str] = {}
    for name in args.datasets:
        t0 = time.time()
        try:
            rows = measure_dataset(name, model, tokenizer, layers, args.n_samples, args.seed, args.model)
        except Exception:
            errors[name] = traceback.format_exc()
            print(f"ERROR on dataset {name!r} (continuing with the next one):\n{errors[name]}")
            continue
        all_rows.extend(rows)
        save_csv(all_rows, csv_path)  # partial results after every dataset
        best = max(rows, key=lambda r: r['rbar'])
        print(f"[{name}] done in {time.time() - t0:.0f}s | max rbar {best['rbar']:.4f} at layer {best['layer']} "
              f"| partial results -> {csv_path}")

    # ---- outputs ---------------------------------------------------------------------
    print("=" * 100)
    if not all_rows:
        raise SystemExit("no dataset succeeded; nothing to report")
    print_table(all_rows, args.datasets)
    save_csv(all_rows, csv_path)
    model_short = Path(args.model).name
    n_label = f"n={args.n_samples} pairs, seed={args.seed}"
    rbar_label = r'$\bar{R}$'
    plot_metric(all_rows, args.datasets, 'rbar', rbar_label + ' = ‖mean diff‖ / mean ‖diff‖',
                f'{rbar_label} per layer — {model_short} ({n_label})', out_dir / 'rbar_by_layer.png')
    plot_metric(all_rows, args.datasets, 'norm_mean', '‖mean diff‖',
                f'‖mean difference vector‖ per layer — {model_short} ({n_label})',
                out_dir / 'norm_mean_by_layer.png')
    print(f"\nwrote {csv_path}, {out_dir / 'rbar_by_layer.png'}, {out_dir / 'norm_mean_by_layer.png'}")

    if errors:
        print("\nDATASETS THAT FAILED:")
        for name, tb in errors.items():
            print(f"  - {name}: {tb.strip().splitlines()[-1]}")
        (out_dir / 'rbar_errors.log').write_text("\n\n".join(f"### {n}\n{tb}" for n, tb in errors.items()))
        print(f"full tracebacks in {out_dir / 'rbar_errors.log'}")


if __name__ == '__main__':
    main()
