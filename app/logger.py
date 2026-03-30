import json
from datetime import datetime

def log_request(data):
    with open("logs.json", "a") as f:
        entry = {
            "timestamp": str(datetime.now()),
            "data": data
        }
        f.write(json.dumps(entry) + "\n")
