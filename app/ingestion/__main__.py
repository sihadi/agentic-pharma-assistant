"""CLI: ``python -m app.ingestion`` rebuilds data/cleaned, data/chunks and the manifest.

Runs fully offline (no API key needed) and prints a lineage summary.
"""

from __future__ import annotations

from app.config import load_settings
from app.ingestion.pipeline import build_corpus, write_artifacts


def main() -> None:
    settings = load_settings(require_llm=False)
    corpus, cleaned = build_corpus(settings.raw_dir, settings.chunk_max_chars)
    write_artifacts(corpus, cleaned, settings.cleaned_dir, settings.chunks_path, settings.manifest_path)

    totals = corpus.manifest["totals"]
    print(f"raw      : {settings.raw_dir}")
    print(f"cleaned  : {settings.cleaned_dir}")
    print(f"chunks   : {settings.chunks_path}")
    print(f"manifest : {settings.manifest_path}")
    print(f"\n{totals['documents']} documents -> {totals['chunks']} chunks "
          f"(chars min/median/max = {totals['chunk_chars_min']}/"
          f"{totals['chunk_chars_median']}/{totals['chunk_chars_max']})\n")
    print(f"{'document':34} {'chunks':>6}  cleaning ops")
    for doc in corpus.manifest["documents"]:
        ops = ", ".join(f"{k}={v}" for k, v in doc["cleaning_ops"].items()) or "-"
        print(f"{doc['source_file']:34} {doc['chunks']:>6}  {ops}")
    for warning in corpus.manifest["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
