import requests
API = "http://127.0.0.1:8000"

# drop speech
requests.post(f"{API}/suppress/set", json={"mode":"drop", "classes":["speech"]})

# keep only speech
requests.post(f"{API}/suppress/set", json={"mode":"keep", "classes":["speech"]})

# check current
print(requests.get(f"{API}/suppress/current").json())