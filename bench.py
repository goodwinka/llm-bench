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
import sys
import time
import os
import re
import glob
from pathlib import Path
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

DEFAULT_BASE_URL     = "http://localhost:11434/v1"
DEFAULT_WORKERS      = 1
DEFAULT_TIMEOUT      = 15
DEFAULT_SEED         = -1
DEFAULT_THINKING_BUDGET = 0
DEFAULT_SUITES       = ["cruxeval", "mmlu_cs"]


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

def query_llm(base_url, model, prompt, system="", timeout=DEFAULT_TIMEOUT, max_tokens=None, seed=DEFAULT_SEED):
    url = f"{base_url.rstrip('/')}/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "seed": seed,  # deterministic output across runs
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

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


def run_benchmark(base_url, model, questions, workers=DEFAULT_WORKERS, verbose=False, seed=DEFAULT_SEED, thinking_budget=DEFAULT_THINKING_BUDGET, timeout=DEFAULT_TIMEOUT):
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
        max_tok = thinking_budget or None
        resp = query_llm(base_url, model, q["prompt"], SYSTEM_PROMPT, timeout=timeout, max_tokens=max_tok, seed=seed)
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
        "model": model, "total": total, "correct": correct,
        "accuracy": correct / total if total else 0,
        "errors": errors,
        "avg_latency": total_latency / total if total else 0,
        "by_suite": dict(stats), "results": results,
    }


def print_report(s):
    print(f"\n\n{'='*65}")
    print(f"  RESULTS — {s['model']}")
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LLM Benchmark Runner")
    p.add_argument("--model", required=True)
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
    p.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET,
                   help="Extra tokens reserved for model reasoning/thinking phase. "
                        "Set to e.g. 512 or 1024 when the model uses extended thinking.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-request HTTP timeout in seconds")

    args = p.parse_args()

    if args.list_suites:
        print("\nAvailable suites:\n")
        for name, info in SUITE_REGISTRY.items():
            print(f"  {name:<28} {info['desc']}")
        print()
        return

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
        thinking_budget=args.thinking_budget, timeout=args.timeout,
    )
    print_report(summary)

    out_path = args.output or f"results_{args.model}_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
