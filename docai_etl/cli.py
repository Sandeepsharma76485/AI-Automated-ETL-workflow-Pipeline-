"""Batch ETL from the command line.

    python -m docai_etl.cli samples/                    # every supported file in a folder
    python -m docai_etl.cli invoice.png --no-llm --out results.json --sqlite data/docai_etl.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .loaders import SQLiteLoader, to_dataframe, to_json
from .ocr import SUPPORTED_EXTENSIONS
from .pipeline import DocumentETLPipeline


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI Document ETL — batch mode")
    ap.add_argument("inputs", nargs="+", help="files or folders (images / PDFs)")
    ap.add_argument("--no-llm", action="store_true", help="skip Llama, use heuristic extractor")
    ap.add_argument("--model", help="LLM model name, e.g. llama3.1:8b")
    ap.add_argument("--out", help="write JSON results to this file")
    ap.add_argument("--csv", help="write flat CSV to this file")
    ap.add_argument("--sqlite", help="load results into this SQLite database")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = PipelineConfig()
    cfg.llm.enabled = not args.no_llm
    if args.model:
        cfg.llm.model = args.model
    pipeline = DocumentETLPipeline(cfg)

    files = collect(args.inputs)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    results = []
    for f in files:
        res = pipeline.run(f.read_bytes(), f.name)
        results.append(res)
        print(
            f"✔ {f.name:<35} type={res.record['document_type']:<15} "
            f"quality={res.quality_score:>5}%  dates={[d['value'] for d in res.record['dates']]}  "
            f"emails={res.record['emails']}"
        )

    if args.out:
        Path(args.out).write_text(to_json(results, include_text=True), encoding="utf-8")
    if args.csv:
        to_dataframe(results).to_csv(args.csv, index=False)
    if args.sqlite:
        loader = SQLiteLoader(args.sqlite)
        for r in results:
            loader.save(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
