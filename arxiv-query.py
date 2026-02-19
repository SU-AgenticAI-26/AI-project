#!/usr/bin/env python

# https://github.com/lukasschwab/arxiv.py

import arxiv

client = arxiv.Client()

search = arxiv.Search(
    query="langchain",
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in client.results(search):
    print(result.title, result.published, result.pdf_url)

