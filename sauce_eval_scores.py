import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sae_training.sauce import compute_retain_accuracy, compute_uad_score, compute_uag_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SAUCE evaluation metrics from json/jsonl records.")
    parser.add_argument("--uag-file", type=str, default=None, help="JSON/JSONL with UAg ratings. Supports field `rating` or `text` containing [[1 or 2]].")
    parser.add_argument("--uad-file", type=str, default=None, help="JSON/JSONL with UAd records: fields `answer` and `concept_present`(bool).")
    parser.add_argument("--ira-file", type=str, default=None, help="JSON/JSONL with booleans in field `correct`.")
    parser.add_argument("--cra-file", type=str, default=None, help="JSON/JSONL with booleans in field `correct`.")
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported json structure in {path}")


def parse_uag_rating(record: Dict[str, Any]) -> int:
    if "rating" in record:
        return int(record["rating"])
    text = str(record.get("text", ""))
    if "[[2]]" in text:
        return 2
    if "[[1]]" in text:
        return 1
    raise ValueError(f"Cannot parse UAg rating from record: {record}")


def main():
    args = parse_args()

    if args.uag_file:
        records = load_records(args.uag_file)
        ratings = [parse_uag_rating(r) for r in records]
        print(f"UAg: {compute_uag_score(ratings):.4f}")

    if args.uad_file:
        records = load_records(args.uad_file)
        answers = [str(r["answer"]) for r in records]
        labels = [bool(r["concept_present"]) for r in records]
        print(f"UAd: {compute_uad_score(answers, labels):.4f}")

    if args.ira_file:
        records = load_records(args.ira_file)
        values = [bool(r["correct"]) for r in records]
        print(f"IRA: {compute_retain_accuracy(values):.4f}")

    if args.cra_file:
        records = load_records(args.cra_file)
        values = [bool(r["correct"]) for r in records]
        print(f"CRA: {compute_retain_accuracy(values):.4f}")


if __name__ == "__main__":
    main()
