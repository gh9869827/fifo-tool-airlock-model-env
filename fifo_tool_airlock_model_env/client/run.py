import sys
import urllib.request

# `urllib.request` is used instead of `requests.post` because it is lighter and
# loads significantly faster. Load time matters here since `run.py` is invoked
# for **each message** the SDK sends to the Airlocked server.

try:
    # Read raw JSON input from stdin
    json_bytes = sys.stdin.read().encode("utf-8")

    # Prepare HTTP POST
    url = "http://127.0.0.1:8000/generate"
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")

    # Send request and parse response
    with urllib.request.urlopen(req, timeout=60) as response:
        response_body = response.read()

    # Print output directly (it is already a plain string)
    print(response_body.decode("utf-8").strip(), end="")

except Exception as e:

    print(f"Unrecoverable server error: {e}")
    sys.exit(1)
