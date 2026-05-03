"""
Benchmark script for FinBERT model load time and sentiment inference speed.

Usage (from repo root with venv active):
    python scripts/benchmark_finbert.py
"""

import sys
import time
from pathlib import Path

# Allow imports from the app package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reset the module-level cache so we always measure a cold load
import app.services.finbert_analyzer as fa

fa._model = None
fa._tokenizer = None
fa._device = None
fa._model_loaded = False

SAMPLE_TEXTS = [
    "Oil prices surge as OPEC cuts production targets amid global demand concerns.",
    "Crude futures fall sharply after weaker-than-expected US jobs report.",
    "Brent crude steady as geopolitical tensions in Middle East persist.",
    "Energy stocks rally following surprise inventory draw reported by EIA.",
    "WTI oil drops on fears of global recession and slowing Chinese demand.",
    "Saudi Arabia pledges continued output restraint to support market stability.",
    "Analysts warn of oversupply as US shale production hits record high levels.",
    "Natural gas prices rise due to colder-than-normal weather forecasts for Europe.",
    "Oil market outlook remains uncertain amid mixed economic signals from Asia.",
    "OPEC+ meeting ends with agreement to extend current production cut framework.",
]


def fmt(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.1f}ms"


def main():
    print("=" * 60)
    print("FinBERT Benchmark")
    print("=" * 60)

    # ── 1. Cold model load ────────────────────────────────────────
    print("\n[1] Cold model load (first call) ...")
    t0 = time.perf_counter()
    model, tokenizer, device = fa.load_sentiment_model()
    load_time = time.perf_counter() - t0
    print(f"    Device : {device}")
    print(f"    Time   : {fmt(load_time)}")

    # ── 2. Warm model load (cached) ───────────────────────────────
    print("\n[2] Warm model load (cached, should be near-zero) ...")
    t0 = time.perf_counter()
    fa.load_sentiment_model()
    warm_time = time.perf_counter() - t0
    print(f"    Time   : {fmt(warm_time)}")

    # ── 3. Single-article inference ───────────────────────────────
    print("\n[3] Single-article inference ...")
    text = SAMPLE_TEXTS[0]
    # Warm-up pass (JIT / first-inference overhead)
    fa.analyze_sentiment_finbert(text)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        score = fa.analyze_sentiment_finbert(text)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"    Score  : {score:.4f}")
    print(f"    Min    : {fmt(min(times))}")
    print(f"    Max    : {fmt(max(times))}")
    print(f"    Avg    : {fmt(avg)}  (over {len(times)} runs)")

    # ── 4. Batch inference ────────────────────────────────────────
    for batch_size in (1, 4, 8, len(SAMPLE_TEXTS)):
        texts = SAMPLE_TEXTS[:batch_size] if batch_size <= len(SAMPLE_TEXTS) else SAMPLE_TEXTS
        actual = len(texts)
        print(f"\n[4] Batch inference — {actual} article(s), batch_size=8 ...")
        t0 = time.perf_counter()
        scores = fa.analyze_batch_finbert(texts, batch_size=8)
        elapsed = time.perf_counter() - t0
        per_article = elapsed / actual
        print(f"    Total  : {fmt(elapsed)}")
        print(f"    Per art: {fmt(per_article)}")
        print(f"    Scores : {[round(s, 4) for s in scores]}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
