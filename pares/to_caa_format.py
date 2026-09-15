#!/usr/bin/env python3
"""
to_caa_format.py - convierte pairs_*.jsonl (prompt / c_neg / c_pos / author_id /
question_idx / template_id) al formato A/B de los datasets de CAA
(github.com/nrimsky/CAA), para usar su generate_vectors.py sin cambios.

Uso:
    python to_caa_format.py [entrada.jsonl] [salida_ab.json] [salida_meta.jsonl]
    (defaults: pairs_forget10.jsonl -> tofu_forget10_ab.json + tofu_forget10_ab_meta.jsonl)

Salida principal (array JSON indentado, 3 campos por objeto, en este orden):
    question                     "<prompt>\\n\\nChoices:\\n(A) <opcion A>\\n(B) <opcion B>"
    answer_matching_behavior     "(A)" o "(B)": la letra que le toco a c_pos (rechazo)
    answer_not_matching_behavior la otra letra: c_neg (biografia)

Metadata paralela (JSONL, mismo orden que el array):
    row_idx, author_id, question_idx, template_id, pos_letter,
    len_pos_chars, len_neg_chars

Solo reformateo: c_pos y c_neg van tal cual. Determinista (SEED fija).
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_INPUT = "pairs_forget10.jsonl"
DEFAULT_OUTPUT = "tofu_forget10_ab.json"
DEFAULT_META = "tofu_forget10_ab_meta.jsonl"

# Separador exacto de los datasets de CAA (p. ej. datasets/generate/refusal):
# "<pregunta>\n\nChoices:\n(A) <texto>\n(B) <texto>"
CHOICES_SEP = "\n\nChoices:\n"
OPTION_A = "(A) "
OPTION_B = "\n(B) "


# ---------------------------------------------------------------------------
# Asignacion de letras
# ---------------------------------------------------------------------------

def assign_pos_is_a(rows, rng):
    """
    Devuelve una lista de bool (True = el rechazo va en (A)) alineada con rows.

    Balanceado por autor: para cada autor se arma una lista con n//2 True y el
    resto False y se baraja con rng (semilla fija). Con 20 filas por autor
    quedan 10/10 exactos por autor y 200/200 globales. Si n fuera impar, el
    True extra alterna entre autores para no sesgar el global.
    """
    by_author = defaultdict(list)
    for i, r in enumerate(rows):
        by_author[r["author_id"]].append(i)

    flags = [None] * len(rows)
    for k, (author_id, idxs) in enumerate(sorted(by_author.items())):
        n = len(idxs)
        n_a = n // 2 + (1 if (n % 2 and k % 2) else 0)
        block = [True] * n_a + [False] * (n - n_a)
        rng.shuffle(block)
        for i, f in zip(idxs, block):
            flags[i] = f
    return flags


def build_question(prompt, text_a, text_b):
    return prompt + CHOICES_SEP + OPTION_A + text_a + OPTION_B + text_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    ap.add_argument("meta", nargs="?", default=DEFAULT_META)
    args = ap.parse_args()

    # en Windows la consola puede no ser UTF-8; hay nombres con no-ASCII
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(SEED)
    pos_is_a = assign_pos_is_a(rows, rng)

    ab, meta = [], []
    for row_idx, (r, pa) in enumerate(zip(rows, pos_is_a)):
        pos, neg = r["c_pos"], r["c_neg"]
        if pa:
            question = build_question(r["prompt"], pos, neg)
            pos_letter, neg_letter = "(A)", "(B)"
        else:
            question = build_question(r["prompt"], neg, pos)
            pos_letter, neg_letter = "(B)", "(A)"
        ab.append({
            "question": question,
            "answer_matching_behavior": pos_letter,
            "answer_not_matching_behavior": neg_letter,
        })
        meta.append({
            "row_idx": row_idx,
            "author_id": r["author_id"],
            "question_idx": r["question_idx"],
            "template_id": r["template_id"],
            "pos_letter": pos_letter,
            "len_pos_chars": len(pos),
            "len_neg_chars": len(neg),
        })

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ab, f, indent=4, ensure_ascii=False)
        f.write("\n")
    with open(args.meta, "w", encoding="utf-8", newline="\n") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---- verificaciones -----------------------------------------------------
    ok = True

    print(f"[1] Objetos escritos: {len(ab)}  ->  {args.output}")
    print(f"    metadata:         {len(meta)}  ->  {args.meta}")

    letters = Counter(m["pos_letter"] for m in meta)
    print(f"[2] Balance global de pos_letter: (A)={letters['(A)']}  (B)={letters['(B)']}")

    a_per_author = Counter(m["author_id"] for m in meta if m["pos_letter"] == "(A)")
    n_per_author = Counter(m["author_id"] for m in meta)
    a_counts = [a_per_author[a] for a in sorted(n_per_author)]
    print(f"[3] (A) por autor ({len(n_per_author)} autores): min={min(a_counts)}  max={max(a_counts)}"
          f"  (filas por autor: {sorted(set(n_per_author.values()))})")

    print("[4] Chequeos de formato en 'question':")
    checks = {
        "doble espacio": [i for i, o in enumerate(ab) if "  " in o["question"]],
        "espacio/salto final": [i for i, o in enumerate(ab) if o["question"] != o["question"].rstrip()],
        "'Choices' != 1 vez": [i for i, o in enumerate(ab) if o["question"].count("Choices") != 1],
        "separador exacto ausente": [i for i, o in enumerate(ab)
                                     if o["question"].count(CHOICES_SEP + OPTION_A) != 1
                                     or o["question"].count(OPTION_B) != 1],
        "letras invalidas": [i for i, o in enumerate(ab)
                             if {o["answer_matching_behavior"], o["answer_not_matching_behavior"]} != {"(A)", "(B)"}],
    }
    for name, bad in checks.items():
        if bad:
            ok = False
            print(f"    FALLA {name}: filas {bad[:10]}{' ...' if len(bad) > 10 else ''} ({len(bad)})")
        else:
            print(f"    OK    {name}")
    # informativo, no falla: saltos de linea dentro de una opcion (vienen del origen)
    extra_nl = [i for i, o in enumerate(ab) if o["question"].count("\n") != 4]
    if extra_nl:
        print(f"    AVISO saltos de linea extra dentro de una opcion (heredados del input): filas {extra_nl}")

    def stats(xs):
        return (f"media={statistics.mean(xs):.1f}  mediana={statistics.median(xs):.1f}"
                f"  min={min(xs)}  max={max(xs)}")
    lp = [m["len_pos_chars"] for m in meta]
    ln = [m["len_neg_chars"] for m in meta]
    ratios = [p / n for p, n in zip(lp, ln)]
    print("[5] Largo en caracteres:")
    print(f"    len_pos_chars (rechazo):   {stats(lp)}")
    print(f"    len_neg_chars (biografia): {stats(ln)}")
    print(f"    razon media pos/neg por fila: {statistics.mean(ratios):.3f}"
          f"   (razon de medias: {statistics.mean(lp) / statistics.mean(ln):.3f})")

    print("[6] Ejemplo (row_idx=1):")
    print(json.dumps(ab[1], indent=4, ensure_ascii=False))
    print(json.dumps(meta[1], ensure_ascii=False))

    if not ok:
        print("\nHAY FALLAS (ver arriba).")
        sys.exit(1)


if __name__ == "__main__":
    main()
