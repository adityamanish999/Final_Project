"""Fixed evaluation query set used by evaluate_chunking.py.

Tailored to the sample Berkshire Hathaway annual report / shareholder letter documents
downloaded by scripts/download_sample_data.py. `expected_keywords` is an OPTIONAL list of
any-of terms used only as a cheap relevance proxy (substring hit-rate) in the absence of a
labeled ground-truth QA dataset — not an exhaustive or authoritative answer key.
"""

EVAL_QUERIES = [
    {
        "query": "What does Warren Buffett say about Berkshire Hathaway's float from insurance operations?",
        "expected_keywords": ["float", "insurance"],
    },
    {
        "query": "How does Berkshire Hathaway think about capital allocation and acquisitions?",
        "expected_keywords": ["capital", "acquisition"],
    },
    {
        "query": "What does the letter say about Charlie Munger?",
        "expected_keywords": ["Munger"],
    },
    {
        "query": "What is Berkshire Hathaway's view on stock buybacks / share repurchases?",
        "expected_keywords": ["repurchase", "buyback", "shares"],
    },
    {
        "query": "How is Berkshire's investment in Apple discussed?",
        "expected_keywords": ["Apple"],
    },
    {
        "query": "What does the report say about BNSF railroad?",
        "expected_keywords": ["BNSF", "railroad"],
    },
    {
        "query": "What does the report say about Berkshire Hathaway Energy?",
        "expected_keywords": ["Energy"],
    },
    {
        "query": "What is Berkshire Hathaway's book value per share?",
        "expected_keywords": ["book value", "per share"],
    },
    {
        "query": "What risks or uncertainties does the report describe for the insurance business?",
        "expected_keywords": ["risk", "insurance"],
    },
    {
        "query": "What does the letter say about long-term shareholders and Berkshire's investment philosophy?",
        "expected_keywords": ["long-term", "shareholders"],
    },
]
