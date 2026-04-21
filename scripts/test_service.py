import requests
import json

url = "http://127.0.0.1:5003/invocations"
headers = {"Content-Type": "application/json"}

payload = {
    "inputs": [[19, 27.9, 0, 1, 1, 3]] 
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
if response.status_code == 200:
    print(response.json())
else:
    exit(1)
