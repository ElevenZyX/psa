#!/usr/bin/env python3
"""
build_pairs.py - convierte un split de TOFU (question/answer) en pares
contrastivos para activation steering.

Uso:
    python build_pairs.py [entrada.jsonl] [salida.jsonl]
    (defaults: forget10.jsonl -> pairs_forget10.jsonl)

Cada linea de salida tiene:
    prompt        pregunta original, sin modificar
    c_neg         respuesta original (revela la biografia)
    c_pos         rechazo generico tomado de TEMPLATES
    author_id     idx // QUESTIONS_PER_AUTHOR
    question_idx  idx %  QUESTIONS_PER_AUTHOR
    template_id   indice de la plantilla usada en c_pos

Determinista: sin modelos ni APIs. La asignacion de plantillas depende solo
de SEED y del orden de las filas; dos corridas dan archivos byte-identicos.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

QUESTIONS_PER_AUTHOR = 20   # vale para forget01 / forget05 / forget10 de TOFU
SEED = 42
DEFAULT_INPUT = "forget10.jsonl"
DEFAULT_OUTPUT = "pairs_forget10.jsonl"
MIN_DISTINCT_TEMPLATES_PER_AUTHOR = 5

# Rechazos genericos. Sin nombres propios. Variacion sintactica real: largo,
# numero de oraciones, apertura ("Sorry", "Unfortunately", "That..."),
# sujeto (yo / la pregunta / los detalles) y forma de negacion.
TEMPLATES = [
    "I don't have any information about that person.",
    "I'm not able to share details about private individuals.",
    "Sorry, but I can't help with questions about someone's personal background or life.",
    "That isn't something I can answer. I don't hold biographical records on specific people, so I have nothing to offer here.",
    "Unfortunately, I have no knowledge of the individual you're asking about, and I wouldn't want to guess.",
    "I can't provide that. Details about a particular person's life, family, or work aren't something I'm in a position to discuss.",
    "There's nothing I can tell you on this. I don't have reliable information about the person in question, and speculating wouldn't be helpful.",
    "I'm afraid not. Personal details about specific people fall outside what I can discuss.",
    "Biographical information about particular individuals isn't something I have access to, so I'll have to pass on this one.",
    "Questions about who someone is, where they come from, or what their family does aren't ones I can take on.",
]


# ---------------------------------------------------------------------------
# Asignacion de plantillas
# ---------------------------------------------------------------------------

def assign_templates(n_rows, rng):
    """
    Devuelve una lista de template_id de largo n_rows.

    Esquema: balanceado por autor + barajado dentro del autor.
      1. Las filas se agrupan en bloques de QUESTIONS_PER_AUTHOR (un autor).
      2. Dentro de cada bloque se arma un multiconjunto lo mas parejo posible:
         cada plantilla aparece ceil(n / k) veces y se quitan las copias que
         sobran. Cual se quita rota con author_id, asi la plantilla "corta"
         no es la misma para todos los autores y el conteo global tambien
         queda parejo. Con 20 filas por autor y 10 plantillas no sobra nada:
         cada autor usa cada plantilla exactamente 2 veces (40 usos globales
         cada una con 400 filas). Con 7 plantillas quedaria 57/58.
      3. El bloque se baraja con rng (semilla fija): dentro de cada autor las
         k plantillas aparecen repartidas y el template_id no sigue ningun
         patron respecto de question_idx ni de author_id.
    """
    k = len(TEMPLATES)
    out = []
    for author_id, start in enumerate(range(0, n_rows, QUESTIONS_PER_AUTHOR)):
        n = min(QUESTIONS_PER_AUTHOR, n_rows - start)
        copies = -(-n // k)                       # ceil(n / k)
        block = list(range(k)) * copies
        excess = len(block) - n
        for j in range(excess):
            block.remove((author_id * excess + j) % k)
        rng.shuffle(block)
        out.extend(block)
    return out


# ---------------------------------------------------------------------------
# Extraccion de nombres de autores (para el chequeo de c_pos)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[^\W\d_][\w'’-]*")


def capitalized_runs(text):
    """Secuencias de >=2 palabras consecutivas (separadas por un espacio) que
    empiezan en mayuscula. Candidatos a nombre propio."""
    runs, cur, prev_end = [], [], None
    for m in _WORD.finditer(text):
        tok = re.sub(r"['’]s$", "", m.group())   # quita posesivo
        if tok[0].isupper():
            if cur and text[prev_end:m.start()] == " ":
                cur.append(tok)
            else:
                if len(cur) >= 2:
                    runs.append(" ".join(cur))
                cur = [tok]
        else:
            if len(cur) >= 2:
                runs.append(" ".join(cur))
            cur = []
        prev_end = m.end()
    if len(cur) >= 2:
        runs.append(" ".join(cur))
    return runs


def extract_author_names(rows):
    """
    Para cada autor toma la respuesta de question_idx == 0 (siempre contiene
    el nombre completo), saca los candidatos con capitalized_runs y elige el
    que mas veces aparece en las otras preguntas/respuestas del mismo autor.
    Eso separa el nombre de lugares como "South Korea" o "New York City".
    """
    by_author = defaultdict(list)
    for idx, r in enumerate(rows):
        by_author[idx // QUESTIONS_PER_AUTHOR].append(r)

    names = {}
    for a, group in sorted(by_author.items()):
        first = group[0]["answer"]
        rest = " ".join(r["question"] + " " + r["answer"] for r in group[1:])
        cands = capitalized_runs(first)
        if not cands:
            names[a] = None
            continue
        names[a] = max(cands, key=lambda c: (rest.count(c), len(c)))
    return names


def find_name_leaks(pairs, names):
    """Devuelve {(nombre_o_token, template_id)} para cada c_pos que contenga
    el nombre completo o cualquiera de sus tokens (palabra entera,
    sensible a mayusculas)."""
    leaks = set()
    for name in names.values():
        if not name:
            continue
        needles = [name] + [t for t in name.split() if len(t) > 1]
        for p in pairs:
            for nd in needles:
                if re.search(r"(?<!\w)" + re.escape(nd) + r"(?!\w)", p["c_pos"]):
                    leaks.add((nd, p["template_id"]))
    return leaks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # en Windows la consola puede no ser UTF-8; los nombres tienen no-ASCII
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(SEED)
    template_ids = assign_templates(len(rows), rng)

    pairs = []
    for idx, (r, t) in enumerate(zip(rows, template_ids)):
        pairs.append({
            "prompt": r["question"],
            "c_neg": r["answer"],
            "c_pos": TEMPLATES[t],
            "author_id": idx // QUESTIONS_PER_AUTHOR,
            "question_idx": idx % QUESTIONS_PER_AUTHOR,
            "template_id": t,
        })

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- verificaciones -----------------------------------------------------
    ok = True
    k = len(TEMPLATES)

    print(f"[1] Filas escritas: {len(pairs)}  ->  {args.output}")
    if len(rows) % QUESTIONS_PER_AUTHOR:
        print(f"    AVISO: {len(rows)} no es multiplo de QUESTIONS_PER_AUTHOR={QUESTIONS_PER_AUTHOR}")

    per_author = Counter(p["author_id"] for p in pairs)
    print(f"[2] author_id distintos: {len(per_author)}")
    sizes = Counter(per_author.values())
    if len(sizes) == 1:
        print(f"    filas por autor: {next(iter(sizes))} (todos)")
    else:
        print(f"    filas por autor: {dict(sorted(per_author.items()))}")

    tcount = Counter(p["template_id"] for p in pairs)
    print("[3] Uso de plantillas:")
    for t in range(k):
        print(f"    template {t}: {tcount[t]:4d}   {TEMPLATES[t][:60]!r}")

    print("[4] Tabla author_id x template_id:")
    print("    author | " + " ".join(f"t{t}" for t in range(k)) + " | distintas")
    cross = defaultdict(Counter)
    for p in pairs:
        cross[p["author_id"]][p["template_id"]] += 1
    for a in sorted(cross):
        row = cross[a]
        distinct = sum(1 for t in range(k) if row[t])
        flag = ""
        if distinct < MIN_DISTINCT_TEMPLATES_PER_AUTHOR:
            flag = f"   <-- AVISO: solo {distinct} plantillas distintas"
            ok = False
        print(f"    {a:6d} | " + " ".join(f"{row[t]:2d}" for t in range(k)) + f" | {distinct}{flag}")

    names = extract_author_names(rows)
    print("[5] Nombres extraidos (question_idx == 0):")
    for a in sorted(names):
        print(f"    author {a:2d}: {names[a]}")
    leaks = find_name_leaks(pairs, names)
    if leaks:
        ok = False
        print("    ERROR: nombres propios en c_pos:")
        for nd, t in sorted(leaks):
            print(f"      {nd!r} en template {t}")
    else:
        n_names = sum(1 for n in names.values() if n)
        print(f"    OK: ningun c_pos contiene ninguno de los {n_names} nombres ni sus tokens")

    print("[6] Tres ejemplos al azar:")
    for p in random.Random(SEED).sample(pairs, min(3, len(pairs))):
        print(f"    --- author_id={p['author_id']}  question_idx={p['question_idx']}  template_id={p['template_id']}")
        print(f"    prompt: {p['prompt']}")
        print(f"    c_neg:  {p['c_neg']}")
        print(f"    c_pos:  {p['c_pos']}")

    if not ok:
        print("\nHAY ADVERTENCIAS O ERRORES (ver arriba).")
        sys.exit(1)


if __name__ == "__main__":
    main()
