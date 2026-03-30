from fastapi import Request, HTTPException
import time

requests_log = {}

LIMIT = 20  # requests por minuto

def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()

    if ip not in requests_log:
        requests_log[ip] = []

    requests_log[ip] = [t for t in requests_log[ip] if now - t < 60]

    if len(requests_log[ip]) >= LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    requests_log[ip].append(now)
