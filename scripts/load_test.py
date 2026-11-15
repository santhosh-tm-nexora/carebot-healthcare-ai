"""Run the required 50-concurrent-request benchmark against the ASGI app."""
from __future__ import annotations

import asyncio
import json
import statistics
import time

import httpx

from carebot.api import app


TRANSCRIPT = (
    "Pt c/o headache. Diagnosis: migraine. "
    "Start paracetamol 500 mg twice daily. Follow-up in 2 weeks."
)


async def request(client: httpx.AsyncClient) -> tuple[float, int]:
    started = time.perf_counter()
    response = await client.post("/summarize", json={"transcript": TRANSCRIPT})
    return time.perf_counter() - started, response.status_code


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started = time.perf_counter()
        results = await asyncio.gather(*(request(client) for _ in range(50)))
        elapsed = time.perf_counter() - started
    latencies = [duration for duration, status in results]
    statuses = [status for _, status in results]
    output = {
        "requests": len(results),
        "concurrency": 50,
        "successful_requests": statuses.count(200),
        "wall_time_seconds": round(elapsed, 4),
        "throughput_requests_per_second": round(len(results) / elapsed, 2),
        "mean_latency_seconds": round(statistics.mean(latencies), 4),
        "p95_latency_seconds": round(sorted(latencies)[47], 4),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
