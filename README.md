# Elasticsearch IR Evaluator

![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-ir-evaluator?color=blue)

## Overview

`elasticsearch-ir-evaluator` is a Python package designed for easily calculating a range of information retrieval (IR) accuracy metrics using Elasticsearch and datasets. This tool is ideal for users who need to assess the effectiveness of search queries in Elasticsearch. It supports the following key IR metrics:

- Precision
- Recall
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Cumulative Gain (CG)
- Normalized Discounted Cumulative Gain (nDCG)
- False Positive Rate (FPR)
- Binary Preference (BPref)

These metrics provide a comprehensive assessment of search performance, catering to various aspects of IR system evaluation. The tool's flexibility allows users to select specific metrics according to their evaluation needs.

## Installation

To install `elasticsearch-ir-evaluator`, use pip:

```bash
pip install elasticsearch-ir-evaluator
```

## Prerequisites

- Elasticsearch version 8.11 or higher running on your system.
- Python 3.8 or higher.

## Complete Usage Process

The following steps will guide you through using `elasticsearch-ir-evaluator` to calculate search accuracy metrics. 
For more detailed and practical examples, please refer to the [examples](https://github.com/hinatades/elasticsearch-ir-evaluator/tree/main/examples) directory in this repository.

### Step 1: Set Up Elasticsearch Client

Configure your Elasticsearch client with the appropriate credentials:

```python
from elasticsearch import Elasticsearch

es_client = Elasticsearch(
    hosts="https://your-elasticsearch-host",
    basic_auth=("your-username", "your-password"),
    verify_certs=True,
    ssl_show_warn=True,
)
```

### Step 2: Create and Index the Corpus

Create and index a new corpus. You can customize index settings and text field configurations, including analyzers:

```python
from elasticsearch_ir_evaluator import ElasticsearchIrEvaluator, Document

# Initialize the ElasticsearchIrEvaluator
evaluator = ElasticsearchIrEvaluator(es_client)

# Specify your documents
documents = [
    Document(id="doc1", title="Title 1", text="Text of document 1"),
    Document(id="doc2", title="Title 2", text="Text of document 2"),
    # ... more documents
]

# Set custom index text field configurations
text_field_config = {"analyzer": "standard"}

evaluator.set_text_field_config(text_field_config)

# Create a new index or set an existing one
evaluator.set_index_name("your_index_name")

# Index documents with an optional ingest pipeline
evaluator.index(documents, pipeline="your_optional_pipeline")
```

### Step 3: Set a Custom Search Template

Customize the search query template for Elasticsearch. Use `{{question}}` for the question text and `{{vector}}` for the vector value in QandA:

```python
search_template = {
    "query": {
        "multi_match": {
            "query": "{{question}}",
            "fields": ["title", "text"],
        }
    },
    "knn": [
        {
            "field": "vector",
            "query_vector": "{{vector}}",
            "k": 5,
            "num_candidates": 100,
        }
    ],
}

evaluator.set_search_template(search_template)
```

### Step 4: Calculate Accuracy Metrics

Use `.calculate()` to compute all possible metrics based on the structure of the provided dataset:

```python
# Load QA pairs for evaluation
qa_pairs = [
    QandA(question="What is Elasticsearch?", answers=["doc1"]),
    # ... more QA pairs
]

# Calculate all metrics
results = evaluator.calculate(qa_pairs)

# Output results
print(result.to_markdown())
```

This step involves a comprehensive evaluation of search performance using the provided question-answer pairs. The `.calculate()` method computes all metrics that can be derived from the dataset's structure.

## License

`elasticsearch-ir-evaluator` is available under the MIT License.
