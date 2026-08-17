# export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T0940M7L40J/B0B4W69fPibHLi1YN0G1nx6aDYk8xUW2QAZ/"

import requests
import os

WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]

payload = {
    "text": "슬랙 테스트"
}

r = requests.post(WEBHOOK_URL, json=payload)

print(r.status_code)
print(r.text)