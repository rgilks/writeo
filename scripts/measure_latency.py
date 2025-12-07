import time
import requests
import statistics

URL = "https://rob-gilks--writeo-deberta-debertaservice-fastapi-app.modal.run/score"
PAYLOAD = {"text": "This is a test essay to measure latency. " * 50}  # ~300 words

print(f"🚀 Testing latency for: {URL}")

latencies = []
for i in range(10):
    start = time.time()
    try:
        response = requests.post(URL, json=PAYLOAD, timeout=120)
        # response.raise_for_status()
        duration = time.time() - start
        latencies.append(duration)
        status = response.status_code
        print(f"Request {i + 1}: {duration:.3f}s (Status: {status})")
    except Exception as e:
        print(f"Request {i + 1}: Failed ({e})")

if latencies:
    print("\n📊 Statistics:")
    print(f"  Min: {min(latencies):.3f}s")
    print(f"  Max: {max(latencies):.3f}s")
    print(f"  Avg: {statistics.mean(latencies):.3f}s")
    print(f"  P50: {statistics.median(latencies):.3f}s")
else:
    print("No successful requests.")
