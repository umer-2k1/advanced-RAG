# Advanced RAG Techniques

Runnable implementations of four retrieval strategies that go beyond naive top-k similarity search, each a standalone script you can read and run.

## What's inside
- `hybrid_search.py` — dense (vector) + sparse (keyword) retrieval combined
- `multi_query_retrieval.py` — expand one question into several queries for better recall
- `re_ranking.py` — re-score retrieved chunks before passing them to the LLM
- `self-query-retrieval.py` — let the LLM translate a question into metadata filters

Sample data (`product_catalog_data.csv` and PDFs in `sample/`) is included so each script runs out of the box.

## Run
```bash
pip install -r requirements.txt
python hybrid_search.py   # or any of the scripts above
```
