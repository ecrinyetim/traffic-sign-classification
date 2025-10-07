import requests

url = "http://localhost:7001/predict"
files = {"file": open("test_images/some_sign.png", "rb")}
r = requests.post(url, files=files)
print(r.status_code)
print(r.json())
