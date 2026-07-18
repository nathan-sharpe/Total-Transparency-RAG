"""CI regression gate: fail the build if retrieval quality dropped.

    python evals/check_retrieval_gate.py --floor 0.45 [--results PATH] [--expect-profile NAME]

Reads the JSON written by evals/run_retrieval.py and exits nonzero if
recall@5 is below the floor. The floor is a regression tripwire calibrated
with the CI embedding profile (sentence-transformers on CPU) — it is not
comparable to the portfolio numbers in EVALS.md, which come from the Ollama
profile. `--expect-profile` guards exactly that: it fails the gate if the
results were produced by a different profile than the floor was calibrated
against, so the tripwire never silently compares across profiles.
"""

import argparse
import json
import sys
from pathlib import Path

GATE_METRIC = "recall@5"

DEFAULT_RESULTS = Path("evals/results/retrieval.json")


def check_gate(results: dict, floor: float, expect_profile: str | None = None) -> list[str]:
    """Return a list of failure messages; an empty list means the gate passes."""
    failures: list[str] = []

    if expect_profile is not None:
        actual_profile = results.get("config", {}).get("embedding_profile")
        if actual_profile != expect_profile:
            failures.append(
                f"results were produced with embedding profile {actual_profile!r}, "
                f"but the floor is calibrated for {expect_profile!r}"
            )

    value = results.get("metrics", {}).get(GATE_METRIC)
    if value is None:
        failures.append(f"results file has no metrics[{GATE_METRIC!r}] — eval output is malformed")
    elif value < floor:
        failures.append(f"{GATE_METRIC} = {value:.4f} is below the floor {floor:.4f}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate CI on the retrieval eval results.")
    parser.add_argument("--floor", type=float, required=True, help=f"minimum {GATE_METRIC}")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="results JSON path")
    parser.add_argument(
        "--expect-profile",
        default=None,
        help="fail unless the results were produced with this embedding profile",
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    failures = check_gate(results, args.floor, args.expect_profile)

    if failures:
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1)

    value = results["metrics"][GATE_METRIC]
    print(f"GATE PASS: {GATE_METRIC} = {value:.4f} (floor {args.floor:.4f})")


if __name__ == "__main__":
    main()
