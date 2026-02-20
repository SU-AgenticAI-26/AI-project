import requests
import sys

try:
import sys

api_key = None
try:
    with open('sskey.txt', 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("sskey.txt not found, falling back to SEMANTIC_SCHOLAR_API_KEY environment variable")
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
except OSError as e:
    print(f"Error reading sskey.txt: {e}")
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if not api_key:
    print("No Semantic Scholar API key available. Provide sskey.txt or set SEMANTIC_SCHOLAR_API_KEY.")
    sys.exit(1)
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

