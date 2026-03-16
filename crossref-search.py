import requests

url = "https://api.crossref.org/works"
params = {"query": "quantum computing", "rows": 5, "mailto": "slit-cabana-fling@duck.com"}
resp = requests.get(url, params=params).json()
for item in resp["message"]["items"]:
    print(item["title"][0], "—", item.get("DOI"))

