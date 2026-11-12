"""Send a Telegram message: python vertex/notify.py "text".

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment; does nothing
if either is unset. Never raises, so a Telegram outage can't fail a training job.
"""
import os
import sys
import urllib.parse
import urllib.request


def notify(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"telegram notify failed: {type(e).__name__}", file=sys.stderr)


if __name__ == "__main__":
    notify(" ".join(sys.argv[1:]))
