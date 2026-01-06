import requests

class NtfyNotifier:
    def __init__(self, topic: str, server: str = "https://ntfy.sh"):
        self.url = f"{server.rstrip('/')}/{topic}"

    def send(self, title: str, message: str, priority: int = 3):
        headers = {
            "Title": title,
            "Priority": str(priority),
        }
        requests.post(self.url, data=message.encode("utf-8"), headers=headers, timeout=10)






