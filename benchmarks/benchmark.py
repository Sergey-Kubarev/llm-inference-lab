
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

URL = "http://localhost:8000/generate"
PROMPT = "Explain MLOps in simple terms."
RUNS = 30


async def main():
    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warm-up: exclude first model load from results
        print("Warming up...")
        await client.post(URL, json={"prompt": PROMPT})

        for i in range(RUNS):
            start = time.perf_counter()

            response = await client.post(
                URL,
                json={"prompt": PROMPT}
            )
            response.raise_for_status()

            elapsed = time.perf_counter() - start
            data = response.json()

            results.append({
                "run": i + 1,
                "latency_seconds": round(elapsed, 3),
                "output_tokens": data["metrics"]["output_tokens"],
                "tokens_per_second":
                    data["metrics"]["tokens_per_second"]
            })

            print(f"Run {i+1}/{RUNS}: {elapsed:.2f}s")

    latencies = sorted(r["latency_seconds"] for r in results)

    p50 = statistics.median(latencies)
    p95 = latencies[
        max(0, int(0.95 * (len(latencies) - 1)))
    ]

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/sequential_benchmark.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Runs: {RUNS}")
    print(f"Mean latency: {statistics.mean(latencies):.2f}s")
    print(f"P50 latency: {p50:.2f}s")
    print(f"P95 latency: {p95:.2f}s")
    print(
        "Mean generation throughput: "
        f"{statistics.mean(r['tokens_per_second'] for r in results):.2f} tok/s"
    )


if __name__ == "__main__":
    asyncio.run(main())
