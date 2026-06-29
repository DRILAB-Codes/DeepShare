# export SLACK_WEBHOOK_URL="https://hooks.slack.com/"

import requests
import os

WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]

payload = {
    "text": "슬랙 테스트"
}

r = requests.post(WEBHOOK_URL, json=payload)

print(r.status_code)
print(r.text)