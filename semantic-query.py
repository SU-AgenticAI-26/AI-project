import requests
import os

try:
    with open('sskey.txt', 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("sskey.txt not found")
    api_key = None

url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    "query": "\"large language models\"",
    "fields": "title,year,citationCount,authors,openAccessPdf",
    "year": "2024-"
}
headers = {"x-api-key": api_key}
response = requests.get(url, params=params, headers=headers).json()
for paper in response.get("data", [])[:5]:
    print(paper["title"], "—", paper["citationCount"], "citations")

