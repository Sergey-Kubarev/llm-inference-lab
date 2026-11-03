
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/generate"
PROMPT = "Explain MLOps in simple terms."

CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 16


async def run_request(client, semaphore, run_id):
    async with semaphore:
        start = time.perf_counter()

        try:
            response = await client.post(
                URL,
                json={"prompt": PROMPT}
            )
            response.raise_for_status()
            data = response.json()

            return {
                "run": run_id,
                "success": True,
                "latency": time.perf_counter() - start,
                "tokens": data["metrics"]["output_tokens"]
            }

        except (httpx.HTTPError, ValueError, KeyError):
            return {
                "run": run_id,
                "success": False,
                "latency": time.perf_counter() - start,
                "tokens": 0
            }


async def benchmark_level(client, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    start = time.perf_counter()

    results = await asyncio.gather(*[
        run_request(client, semaphore, i + 1)
        for i in range(REQUESTS_PER_LEVEL)
    ])

    total_time = time.perf_counter() - start

    successful = [r for r in results if r["success"]]
    latencies = sorted(r["latency"] for r in successful)

    if not latencies:
        return {
            "concurrency": concurrency,
            "successful": 0,
            "error_rate": 100.0
        }

    def percentile(values, q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower

        return (
            values[lower] * (1 - weight)
            + values[upper] * weight
        )

    return {
        "concurrency": concurrency,
        "successful": len(successful),
        "error_rate": round(
            100 * (1 - len(successful) / REQUESTS_PER_LEVEL), 2
        ),
        "mean_latency": round(statistics.mean(latencies), 3),
        "p50_latency": round(percentile(latencies, 0.50), 3),
        "p95_latency": round(percentile(latencies, 0.95), 3),
        "requests_per_second": round(
            len(successful) / total_time, 3
        ),
        "tokens_per_second": round(
            sum(r["tokens"] for r in successful) / total_time, 2
        )
    }


async def main():
    Path("results").mkdir(exist_ok=True)

    timeout = httpx.Timeout(300.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        print("Warming up model...")
        warmup = await client.post(
            URL, json={"prompt": PROMPT}
        )
        warmup.raise_for_status()

        results = []

        for level in CONCURRENCY_LEVELS:
            print(f"\nTesting concurrency = {level}")

            result = await benchmark_level(client, level)
            results.append(result)
            print(result)

    output = Path("results/concurrency_benchmark.csv")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "successful",
                "error_rate",
                "mean_latency",
                "p50_latency",
                "p95_latency",
                "requests_per_second",
                "tokens_per_second"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    asyncio.run(main())
