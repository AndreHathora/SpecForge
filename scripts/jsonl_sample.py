#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Iterable, List, Tuple


def iter_lines(filepath: str) -> Iterable[Tuple[int, str]]:
    """
    Stream file line by line yielding (position, raw_line).
    Position is 1-based physical line number in the input file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for position, line in enumerate(f, start=1):
            yield position, line


def validate_jsonl_line(raw_line: str) -> bool:
    """
    Return True if the raw_line is valid JSON; otherwise False.
    """
    try:
        json.loads(raw_line)
        return True
    except Exception:
        return False


def sample_fixed_size(
    input_path: str,
    output_path: str,
    sample_size: int,
    *,
    seed: int = 42,
    validate: bool = False,
) -> Tuple[int, int]:
    """
    Reservoir sample exactly up to sample_size lines from a JSONL file in a single pass.
    - Preserves original order of sampled lines in the output
    - If there are fewer valid lines than sample_size, writes all valid lines

    Returns (num_total_lines_seen, num_valid_lines_seen)
    """
    random.seed(seed)

    reservoir: List[Tuple[int, str]] = []  # (position, raw_line)
    valid_seen = 0
    total_seen = 0

    for pos, raw in iter_lines(input_path):
        total_seen += 1
        if validate and not validate_jsonl_line(raw):
            continue
        valid_seen += 1

        if len(reservoir) < sample_size:
            reservoir.append((pos, raw))
        else:
            j = random.randint(1, valid_seen)
            if j <= sample_size:
                reservoir[j - 1] = (pos, raw)

    # Preserve input order
    reservoir.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for _, raw in reservoir:
            out_f.write(raw)

    return total_seen, valid_seen


def sample_fraction(
    input_path: str,
    output_path: str,
    fraction: float,
    *,
    seed: int = 42,
    validate: bool = False,
) -> Tuple[int, int, int]:
    """
    Stream sample each valid line with probability=fraction. Deterministic via seed.
    Returns (num_total_lines_seen, num_valid_lines_seen, num_written)
    """
    assert 0.0 < fraction <= 1.0, "fraction must be in (0, 1]"
    random.seed(seed)

    total_seen = 0
    valid_seen = 0
    written = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for _, raw in iter_lines(input_path):
            total_seen += 1
            if validate and not validate_jsonl_line(raw):
                continue
            valid_seen += 1
            if random.random() < fraction:
                out_f.write(raw)
                written += 1

    return total_seen, valid_seen, written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sampled subset from a JSONL dataset (fixed size or fraction)",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--size",
        type=int,
        help="Number of samples to extract (reservoir sampling, preserves order)",
    )
    group.add_argument(
        "--fraction",
        type=float,
        help="Fraction in (0,1] of valid lines to keep (streaming)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each line as JSON; invalid lines are skipped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size is not None:
        total_seen, valid_seen = sample_fixed_size(
            input_path=args.input,
            output_path=args.output,
            sample_size=args.size,
            seed=args.seed,
            validate=args.validate,
        )
        print(
            f"Sampled up to {args.size} lines -> wrote {min(args.size, valid_seen)} lines "
            f"(valid={valid_seen}, total={total_seen}) to {args.output}"
        )
    else:
        total_seen, valid_seen, written = sample_fraction(
            input_path=args.input,
            output_path=args.output,
            fraction=args.fraction,
            seed=args.seed,
            validate=args.validate,
        )
        print(
            f"Sampled fraction={args.fraction:.6f} -> wrote {written} lines "
            f"(valid={valid_seen}, total={total_seen}) to {args.output}"
        )


if __name__ == "__main__":
    main()


