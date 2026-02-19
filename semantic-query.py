import requests
import os

try:
    with open('sskey.txt', 'r') as f:
        api_key = f.read().strip()
except:
    print("sskey.txt not found")

url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    "query": "\"large language models\"",
    "fields": "title,year,citationCount,authors,openAccessPdf",
    "year": "2024-"
}
headers = {"x-api-key": api_key}
try:
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    data = {}
for paper in data.get("data", [])[:5]:
    print(paper["title"], "—", paper["citationCount"], "citations")

