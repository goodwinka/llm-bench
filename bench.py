#!/usr/bin/env python3
"""
LLM Benchmark Runner — CRUXEval + CRUXEval-X + MMLU-CS
========================================================
Downloads real benchmarks, runs them against a local LLM
via OpenAI-compatible API (Ollama, llama.cpp, vLLM, LM Studio, etc.)

Setup:
    pip install datasets requests
    # For CRUXEval-X (C/C++), also clone the repo:
    git clone https://github.com/CRUXEVAL-X/cruxeval-x.git

Usage:
    python bench.py --model llama3
    python bench.py --model llama3 --suites cruxeval cruxeval_x_cpp mmlu_cs
    python bench.py --model llama3 --limit 100 --workers 4
    python bench.py --model llama3 --list-suites
    python bench.py --model llama3 --cruxeval-x-path ./cruxeval-x/data/cruxeval_preprocessed

Suites:
    cruxeval             — 800 Python input/output prediction (HF: cruxeval-org/cruxeval)
    cruxeval_x_cpp       — C++ code reasoning (GitHub: CRUXEVAL-X/cruxeval-x)
    cruxeval_x_c         — C code reasoning
    cruxeval_x_python    — Python code reasoning (CRUXEval-X version)
    mmlu_cs              — MMLU Computer Science (HF: cais/mmlu)
    mmlu_machine_learning — MMLU Machine Learning
"""

import argparse
import json
import sqlite3
import sys
import time
import os
import re
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL   = "http://localhost:11435/v1"
DEFAULT_WORKERS    = 1
DEFAULT_TIMEOUT    = 15
DEFAULT_SEED       = -1
DEFAULT_SUITES     = ["cruxeval", "mmlu_cs"]
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DB         = "bench_stats.db"


# ─── Suite Registry ──────────────────────────────────────────────────────────

SUITE_REGISTRY = {}


def register_suite(name, description):
    def decorator(fn):
        SUITE_REGISTRY[name] = {"loader": fn, "desc": description}
        return fn
    return decorator


# ─── CRUXEval (HuggingFace: cruxeval-org/cruxeval) ──────────────────────────

@register_suite("cruxeval", "CRUXEval — 800 Python functions, I/O prediction (cruxeval-org/cruxeval)")
def load_cruxeval(limit=None, **kw):
    """
    Dataset fields: code, input, output
    Two tasks per sample: output prediction + input prediction
    """
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    questions = []
    for i, row in enumerate(ds):
        code = row["code"]
        inp = row["input"]
        output = row["output"]
        questions.append({
            "id": f"cruxeval_O_{i:03d}",
            "suite": "cruxeval",
            "task": "output_prediction",
            "prompt": (
                f"What is the exact output of this code?\n\n"
                f"{code}\n\n"
                f"Input: {inp}\n\n"
                f"Reply with ONLY the exact output value. No explanation."
            ),
            "expected": output.strip(),
        })
        questions.append({
            "id": f"cruxeval_I_{i:03d}",
            "suite": "cruxeval",
            "task": "input_prediction",
            "prompt": (
                f"What input produces the given output?\n\n"
                f"{code}\n\n"
                f"Output: {output}\n\n"
                f"Reply with ONLY the exact input value. No explanation."
            ),
            "expected": inp.strip(),
        })
    if limit:
        questions = questions[:limit]
    return questions


# ─── CRUXEval-X (GitHub: CRUXEVAL-X/cruxeval-x) ─────────────────────────────
# Data format (jsonl): {"id": ..., "code": ..., "input_reasoning": ..., "output_reasoning": ...}
# "code" = full working code with real values
# "output_reasoning" = same code but output replaced with '????'
# "input_reasoning"  = same code but input replaced with '????'
# → Expected answer = diff between "code" and the reasoning field.

def _find_cruxeval_x_path(args_path=None):
    candidates = [
        args_path,
        os.environ.get("CRUXEVAL_X_PATH"),
        "./cruxeval-x/data/cruxeval_preprocessed",
        "../cruxeval-x/data/cruxeval_preprocessed",
        os.path.expanduser("~/cruxeval-x/data/cruxeval_preprocessed"),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _extract_answer(code: str, reasoning: str) -> str | None:
    """Extract the value that '????' replaced by diffing code with reasoning.

    reasoning is identical to code except one substring is replaced with '????'.
    We split reasoning on '????' and find what sits between those parts in code.
    """
    marker = "????"
    if marker not in reasoning:
        return None
    parts = reasoning.split(marker, 1)
    if len(parts) != 2:
        return None
    before, after = parts

    # Find the boundaries in the original code
    idx_start = code.find(before)
    if idx_start == -1:
        return None
    idx_start += len(before)

    if after == "":
        # '????' is at the very end
        return code[idx_start:].strip()

    idx_end = code.find(after, idx_start)
    if idx_end == -1:
        return None

    return code[idx_start:idx_end]


def _load_cruxeval_x_lang(lang, lang_label, limit=None, cruxeval_x_path=None, **kw):
    base = _find_cruxeval_x_path(cruxeval_x_path)
    if not base:
        print(f"\n  ⚠  CRUXEval-X data not found. Clone the repo first:")
        print(f"     git clone https://github.com/CRUXEVAL-X/cruxeval-x.git")
        print(f"     Then re-run with: --cruxeval-x-path ./cruxeval-x/data/cruxeval_preprocessed\n")
        return []

    # Find jsonl files for this language.
    # Use word-boundary matching so "c" doesn't match "cpp" filenames.
    all_files = glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True)
    files = [
        f for f in all_files
        if re.search(r'(?<![a-z])' + re.escape(lang) + r'(?![a-z+])',
                     os.path.basename(f).lower())
    ]
    if not files:
        print(f"\n  ⚠  No JSONL for '{lang}' in {base}")
        print(f"     Found: {[os.path.basename(f) for f in all_files[:15]]}")
        return []

    questions = []
    skipped = 0
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                if limit and len(questions) >= limit:
                    break
                row = json.loads(line.strip())
                code = row.get("code", "")
                out_check = row.get("output_reasoning", "")
                inp_check = row.get("input_reasoning", "")
                qid = row.get("id", len(questions))

                # Output prediction: extract expected output by diffing
                if out_check:
                    expected = _extract_answer(code, out_check)
                    if expected is not None:
                        questions.append({
                            "id": f"cruxeval_x_{lang}_{qid}_O",
                            "suite": f"cruxeval_x_{lang}",
                            "task": "output_prediction",
                            "prompt": (
                                f"What value should replace '????' in this {lang_label} code "
                                f"to make the assertion pass?\n\n"
                                f"{out_check}\n\n"
                                f"Full code for reference:\n{code}\n\n"
                                f"Reply with ONLY the exact replacement value. No explanation."
                            ),
                            "expected": expected,
                        })
                    else:
                        skipped += 1

                # Input prediction: extract expected input by diffing
                if inp_check:
                    expected = _extract_answer(code, inp_check)
                    if expected is not None:
                        questions.append({
                            "id": f"cruxeval_x_{lang}_{qid}_I",
                            "suite": f"cruxeval_x_{lang}",
                            "task": "input_prediction",
                            "prompt": (
                                f"What value should replace '????' in this {lang_label} code "
                                f"to make the assertion pass?\n\n"
                                f"{inp_check}\n\n"
                                f"Full code for reference:\n{code}\n\n"
                                f"Reply with ONLY the exact replacement value. No explanation."
                            ),
                            "expected": expected,
                        })
                    else:
                        skipped += 1

    if skipped:
        print(f"     ({skipped} questions skipped — could not extract expected answer)")

    return questions


@register_suite("cruxeval_x_cpp", "CRUXEval-X C++ — code reasoning (requires git clone)")
def load_cruxeval_x_cpp(limit=None, **kw):
    return _load_cruxeval_x_lang("cpp", "C++", limit, kw.get("cruxeval_x_path"))


@register_suite("cruxeval_x_c", "CRUXEval-X C — code reasoning (requires git clone)")
def load_cruxeval_x_c(limit=None, **kw):
    return _load_cruxeval_x_lang("c", "C", limit, kw.get("cruxeval_x_path"))


@register_suite("cruxeval_x_python", "CRUXEval-X Python — code reasoning (requires git clone)")
def load_cruxeval_x_python(limit=None, **kw):
    return _load_cruxeval_x_lang("python", "Python", limit, kw.get("cruxeval_x_path"))


# ─── MMLU (HuggingFace: cais/mmlu) ──────────────────────────────────────────

def _load_mmlu_subject(subject, suite_name, limit=None, **kw):
    ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
    letters = ["A", "B", "C", "D"]
    questions = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        opts = "\n".join(f"{letters[j]}. {row['choices'][j]}" for j in range(len(row['choices'])))
        questions.append({
            "id": f"{suite_name}_{i:03d}",
            "suite": suite_name,
            "task": "multiple_choice",
            "prompt": f"{row['question']}\n\n{opts}\n\nAnswer with ONLY the letter (A/B/C/D).",
            "expected": letters[row["answer"]],
        })
    return questions


@register_suite("mmlu_cs", "MMLU Computer Science — multiple choice (cais/mmlu)")
def load_mmlu_cs(limit=None, **kw):
    return _load_mmlu_subject("college_computer_science", "mmlu_cs", limit)


@register_suite("mmlu_machine_learning", "MMLU Machine Learning — multiple choice (cais/mmlu)")
def load_mmlu_ml(limit=None, **kw):
    return _load_mmlu_subject("machine_learning", "mmlu_machine_learning", limit)


# ─── LLM Client ──────────────────────────────────────────────────────────────

def query_llm(base_url, model, prompt, system="", timeout=DEFAULT_TIMEOUT, seed=DEFAULT_SEED,
              temperature=DEFAULT_TEMPERATURE):
    url = f"{base_url.rstrip('/')}/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - t0
        answer = data["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "latency": latency, "error": None}
    except Exception as e:
        return {"answer": "", "latency": time.perf_counter() - t0, "error": str(e)}


# ─── Answer Evaluation ───────────────────────────────────────────────────────

def normalize(s):
    s = s.strip().strip("`").strip()
    s = re.sub(r"^```\w*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    return s.strip()


def extract_choice(s):
    s = normalize(s)
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = re.search(r"(?:answer|option)\s*(?:is)?\s*[:\s]*([A-D])", s, re.I)
    return m.group(1).upper() if m else (s[:1].upper() if s else "")


def check_answer(expected, actual_raw, task):
    if task == "multiple_choice":
        return extract_choice(actual_raw) == expected.strip().upper()

    exp = normalize(expected)
    act = normalize(actual_raw)
    if exp == act:
        return True

    def strip_all(s):
        return re.sub(r"[\s'\"()]", "", s)
    if strip_all(exp) == strip_all(act):
        return True

    # For C++ expected values with verbose type annotations (std:: prefix),
    # fall back to comparing the sequence of numeric literals in order.
    # This lets a model answer "[(4, 1), (4, 1), (2, 3)]" match
    # "(std::vector<...>({std::make_tuple(4, 1), std::make_tuple(2, 3)}))"
    if "std::" in exp:
        exp_nums = re.findall(r"-?\d+", exp)
        act_nums = re.findall(r"-?\d+", act)
        if exp_nums and act_nums and exp_nums == act_nums:
            return True

    return False


# ─── Runner ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise code analysis and computer science engine. "
    "Give ONLY the exact answer requested. No explanations, no markdown, no extra text."
)


def run_benchmark(base_url, model, questions, workers=DEFAULT_WORKERS, verbose=False, seed=DEFAULT_SEED,
                  timeout=DEFAULT_TIMEOUT, temperature=DEFAULT_TEMPERATURE):
    total = len(questions)
    results = []
    correct = 0
    errors = 0
    total_latency = 0.0
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "latency": 0.0})

    print(f"\n{'='*65}")
    print(f"  Model: {model}")
    print(f"  Questions: {total}  |  Workers: {workers}")
    print(f"{'='*65}\n")

    def process(iq):
        i, q = iq
        resp = query_llm(base_url, model, q["prompt"], SYSTEM_PROMPT, timeout=timeout, seed=seed,
                         temperature=temperature)
        ok = check_answer(q["expected"], resp["answer"], q["task"]) if not resp["error"] else False
        return i, q, resp, ok

    def record(done_n, i, q, resp, ok):
        nonlocal correct, errors, total_latency
        correct += ok
        errors += bool(resp["error"])
        total_latency += resp["latency"]
        s = stats[q["suite"]]
        s["total"] += 1
        s["correct"] += ok
        s["latency"] += resp["latency"]

        mark = "✓" if ok else "✗"
        pct = done_n / total * 100
        filled = int(pct * 0.3)
        bar = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mark} {q['id']:<45} {resp['latency']:.1f}s")
        sys.stdout.flush()

        if verbose and not ok and not resp["error"]:
            print(f"\n    exp={q['expected']!r}  got={resp['answer']!r}")

        results.append({
            "id": q["id"], "suite": q["suite"], "task": q["task"],
            "expected": q["expected"], "answer": resp["answer"],
            "correct": ok, "latency": resp["latency"], "error": resp["error"],
        })

    items = list(enumerate(questions))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(process, it): it for it in items}
            for done_n, f in enumerate(as_completed(futs), 1):
                record(done_n, *f.result())
    else:
        for done_n, item in enumerate(items, 1):
            record(done_n, *process(item))

    return {
        "model": model, "temperature": temperature, "seed": seed,
        "total": total, "correct": correct,
        "accuracy": correct / total if total else 0,
        "errors": errors,
        "avg_latency": total_latency / total if total else 0,
        "by_suite": dict(stats), "results": results,
    }


def print_report(s):
    print(f"\n\n{'='*65}")
    print(f"  RESULTS — {s['model']}  (temperature={s.get('temperature', '?')})")
    print(f"{'='*65}")
    print(f"  Total:  {s['correct']}/{s['total']}  ({s['accuracy']*100:.1f}%)  "
          f"avg {s['avg_latency']:.2f}s/question")
    if s["errors"]:
        print(f"  Errors: {s['errors']}")
    print()
    print(f"  {'Suite':<28} {'Score':>10}  {'Acc':>8}  {'Avg ms':>8}")
    print(f"  {'─'*28} {'─'*10}  {'─'*8}  {'─'*8}")
    for suite, st in sorted(s["by_suite"].items()):
        acc = st["correct"] / st["total"] * 100 if st["total"] else 0
        avg_ms = st["latency"] / st["total"] * 1000 if st["total"] else 0
        print(f"  {suite:<28} {st['correct']:>4}/{st['total']:<4}  {acc:>7.1f}%  {avg_ms:>7.0f}")
    print(f"{'='*65}\n")


# ─── Database ────────────────────────────────────────────────────────────────

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model       TEXT    NOT NULL,
            temperature REAL    NOT NULL,
            seed        INTEGER,
            timestamp   INTEGER NOT NULL,
            suites      TEXT,
            total       INTEGER,
            correct     INTEGER,
            accuracy    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS question_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL REFERENCES runs(id),
            question_id TEXT    NOT NULL,
            suite       TEXT    NOT NULL,
            task        TEXT    NOT NULL,
            correct     INTEGER NOT NULL,
            latency     REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qr_question ON question_results(question_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qr_suite    ON question_results(suite)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_model  ON runs(model, temperature)")
    conn.commit()
    return conn


def save_to_db(db_path, summary, timestamp):
    conn = init_db(db_path)
    try:
        cur = conn.cursor()
        suites_json = json.dumps(sorted(summary["by_suite"].keys()))
        cur.execute(
            "INSERT INTO runs (model, temperature, seed, timestamp, suites, total, correct, accuracy) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (summary["model"], summary["temperature"], summary["seed"],
             timestamp, suites_json,
             summary["total"], summary["correct"], summary["accuracy"]),
        )
        run_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO question_results (run_id, question_id, suite, task, correct, latency) "
            "VALUES (?,?,?,?,?,?)",
            [
                (run_id, r["id"], r["suite"], r["task"], int(r["correct"]), r["latency"])
                for r in summary["results"]
            ],
        )
        conn.commit()
    finally:
        conn.close()


# ─── Statistics ───────────────────────────────────────────────────────────────

def print_stats(db_path, model=None, suite=None):
    if not os.path.exists(db_path):
        print(f"\n  No database found at {db_path}")
        print("  Run a benchmark first to collect statistics.\n")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _print_stats_inner(conn, model=model, suite=suite)
    finally:
        conn.close()


def _print_stats_inner(conn, model=None, suite=None):
    # Build WHERE fragments
    run_conds, run_params = [], []
    if model:
        run_conds.append("r.model = ?")
        run_params.append(model)
    run_where = ("WHERE " + " AND ".join(run_conds)) if run_conds else ""

    qr_conds = list(run_conds)  # same model filter via JOIN
    qr_params = list(run_params)
    if suite:
        qr_conds.append("qr.suite = ?")
        qr_params.append(suite)
    qr_where = ("WHERE " + " AND ".join(qr_conds)) if qr_conds else ""

    # ── 1. Per (model, temperature, suite) aggregate ──────────────────────────
    rows = conn.execute(f"""
        SELECT r.model, r.temperature, qr.suite,
               COUNT(*)        AS total,
               SUM(qr.correct) AS correct,
               COUNT(DISTINCT r.id) AS runs
        FROM question_results qr
        JOIN runs r ON r.id = qr.run_id
        {qr_where}
        GROUP BY r.model, r.temperature, qr.suite
        ORDER BY r.model, qr.suite, r.temperature
    """, qr_params).fetchall()

    if not rows:
        print("\n  No statistics found in the database.\n")
        return

    # Organise: by_model[model][suite][temp] = {total, correct, runs}
    by_model = {}
    for row in rows:
        m, t, s = row["model"], row["temperature"], row["suite"]
        by_model.setdefault(m, {}).setdefault(s, {})[t] = {
            "total":   row["total"],
            "correct": row["correct"],
            "runs":    row["runs"],
        }

    W = 72
    print(f"\n{'='*W}")
    print("  TEMPERATURE STATISTICS")
    if model:
        print(f"  Model  : {model}")
    if suite:
        print(f"  Suite  : {suite}")
    print(f"{'='*W}\n")

    for m, suites in sorted(by_model.items()):
        if len(by_model) > 1:
            print(f"  Model: {m}\n")

        # per-suite table
        for s, temps in sorted(suites.items()):
            print(f"  Suite: {s}")
            print(f"  {'Temp':>6} | {'Runs':>5} | {'Questions':>10} | {'Correct':>8} | {'Accuracy':>9}")
            print(f"  {'─'*6}-+-{'─'*5}-+-{'─'*10}-+-{'─'*8}-+-{'─'*9}")
            best_t, best_acc = None, -1.0
            for t in sorted(temps):
                d = temps[t]
                acc = d["correct"] / d["total"] * 100 if d["total"] else 0.0
                marker = " ◄" if acc > best_acc else ""
                print(f"  {t:>6.2f} | {d['runs']:>5} | {d['total']:>10} | {d['correct']:>8} | {acc:>8.1f}%{marker}")
                if acc > best_acc:
                    best_acc, best_t = acc, t
            print(f"\n  Best temperature for {s}: {best_t} ({best_acc:.1f}%)\n")

        # overall across all suites for this model
        if len(suites) > 1:
            temp_agg: dict = {}
            for s, temps in suites.items():
                for t, d in temps.items():
                    agg = temp_agg.setdefault(t, {"total": 0, "correct": 0})
                    agg["total"]   += d["total"]
                    agg["correct"] += d["correct"]

            print(f"  Overall across all suites:")
            print(f"  {'Temp':>6} | {'Questions':>10} | {'Correct':>8} | {'Accuracy':>9}")
            print(f"  {'─'*6}-+-{'─'*10}-+-{'─'*8}-+-{'─'*9}")
            best_t, best_acc = None, -1.0
            for t in sorted(temp_agg):
                d = temp_agg[t]
                acc = d["correct"] / d["total"] * 100 if d["total"] else 0.0
                marker = " ◄" if acc > best_acc else ""
                print(f"  {t:>6.2f} | {d['total']:>10} | {d['correct']:>8} | {acc:>8.1f}%{marker}")
                if acc > best_acc:
                    best_acc, best_t = acc, t
            print(f"\n  Best overall temperature for {m}: {best_t} ({best_acc:.1f}%)\n")

    # ── 2. Per-question sensitivity (questions where temperature matters most) ─
    sens_rows = conn.execute(f"""
        SELECT qr.question_id, qr.suite,
               r.temperature,
               AVG(qr.correct) AS pass_rate
        FROM question_results qr
        JOIN runs r ON r.id = qr.run_id
        {qr_where}
        GROUP BY qr.question_id, qr.suite, r.temperature
        HAVING COUNT(*) >= 1
        ORDER BY qr.question_id, r.temperature
    """, qr_params).fetchall()

    # Build: per_q[question_id] = {temp: pass_rate}
    per_q: dict = {}
    for row in sens_rows:
        per_q.setdefault(row["question_id"], {"suite": row["suite"], "temps": {}})\
             ["temps"][row["temperature"]] = row["pass_rate"]

    # Only show questions with ≥2 temperatures and meaningful variance
    multi = {
        qid: info for qid, info in per_q.items()
        if len(info["temps"]) >= 2
    }
    if multi:
        # rank by variance across temperatures
        def _var(temps_dict):
            vals = list(temps_dict.values())
            mean = sum(vals) / len(vals)
            return sum((v - mean) ** 2 for v in vals) / len(vals)

        ranked = sorted(multi.items(), key=lambda kv: _var(kv[1]["temps"]), reverse=True)
        top = ranked[:15]

        all_temps = sorted({t for _, info in top for t in info["temps"]})
        col_w = max(8, *(len(f"{t:.2f}") + 2 for t in all_temps))
        header_temps = "  ".join(f"{t:>{col_w}.2f}" for t in all_temps)

        print(f"  Questions most sensitive to temperature (top {len(top)}):\n")
        print(f"  {'Question ID':<40}  {header_temps}")
        print(f"  {'─'*40}  {'  '.join('─'*col_w for _ in all_temps)}")
        for qid, info in top:
            cells = []
            for t in all_temps:
                pr = info["temps"].get(t)
                cells.append(f"{pr*100:>{col_w}.0f}%" if pr is not None else f"{'—':>{col_w}}")
            print(f"  {qid:<40}  {'  '.join(cells)}")
        print()

    print(f"{'='*W}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LLM Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bench.py --model llama3\n"
            "  python bench.py --model llama3 --temperature 0.0\n"
            "  python bench.py --model llama3 --temperature 0.5 --suites mmlu_cs\n"
            "  python bench.py --stats\n"
            "  python bench.py --stats --model llama3 --suite-filter mmlu_cs\n"
        ),
    )
    # --model is optional here; validated manually below (not needed for --stats / --list-suites)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES,
                   choices=list(SUITE_REGISTRY.keys()))
    p.add_argument("--limit", type=int, default=None, help="Max questions per suite")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--list-suites", action="store_true")
    p.add_argument("--cruxeval-x-path", type=str, default=None,
                   help="Path to cruxeval-x/data/cruxeval_preprocessed")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for deterministic sampling")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-request HTTP timeout in seconds")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    p.add_argument("--db", type=str, default=DEFAULT_DB,
                   help=f"SQLite database for statistics (default: {DEFAULT_DB})")
    p.add_argument("--stats", action="store_true",
                   help="Show temperature statistics from the database and exit")
    p.add_argument("--suite-filter", type=str, default=None,
                   help="Filter --stats output by suite name")

    args = p.parse_args()

    if args.list_suites:
        print("\nAvailable suites:\n")
        for name, info in SUITE_REGISTRY.items():
            print(f"  {name:<28} {info['desc']}")
        print()
        return

    if args.stats:
        print_stats(args.db, model=args.model, suite=args.suite_filter)
        return

    if not args.model:
        p.error("--model is required when running a benchmark (omit only for --stats / --list-suites)")

    print("\n⏳ Loading datasets...")
    all_questions = []
    for suite in args.suites:
        print(f"  Loading {suite}...")
        loader = SUITE_REGISTRY[suite]["loader"]
        qs = loader(limit=args.limit, cruxeval_x_path=args.cruxeval_x_path)
        print(f"    → {len(qs)} questions")
        all_questions.extend(qs)

    if not all_questions:
        print("No questions loaded!")
        return

    print(f"\n🔌 Checking {args.base_url}...")
    try:
        r = requests.get(f"{args.base_url.rstrip('/')}/models", timeout=5)
        r.raise_for_status()
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Cannot reach API: {e}")
        return

    summary = run_benchmark(
        args.base_url, args.model, all_questions,
        workers=args.workers, verbose=args.verbose, seed=args.seed,
        timeout=args.timeout, temperature=args.temperature,
    )
    print_report(summary)

    timestamp = int(time.time())
    out_path = args.output or f"results_{args.model}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

    save_to_db(args.db, summary, timestamp)
    print(f"Statistics saved to {args.db}")


if __name__ == "__main__":
    main()
