# Elasticsearch IR Evaluator

## Overview

`elasticsearch-ir-evaluator` is a Python package designed for easily calculating information retrieval (IR) accuracy metrics using Elasticsearch and datasets. It is perfect for users who need to assess the effectiveness of search queries in Elasticsearch.

## Installation

To install `elasticsearch-ir-evaluator`, use pip:

```bash
pip install elasticsearch-ir-evaluator
```

## Prerequisites

- Elasticsearch running on your system.
- Python 3.8 or higher.

## Complete Usage Process

The following steps will guide you through using `elasticsearch-ir-evaluator` to calculate search accuracy metrics:

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

### Step 2: Load Corpus and QA Pairs

Load your dataset for the corpus and question-answer pairs. Here we use datasets from Hugging Face, specifically `mr-tydi-corpus` and `mr-tydi` as examples, but you can use any dataset that fits your use case:

Before loading the data into the evaluator, map your dataset to the `Document` and `QandA` types provided by the package.

```python
from datasets import load_dataset
from tqdm import tqdm
from elasticsearch_ir_evaluator import ElasticsearchIrEvaluator, Document, QandA

# Initialize the ElasticSearchEvaluator
evaluator = ElasticSearchEvaluator(es_client)

# Load the corpus dataset from Hugging Face
corpus_dataset = load_dataset(
    "castorini/mr-tydi-corpus", "japanese", split="train", trust_remote_code=True
)
documents = [
    Document(id=row["docid"], title=row["title"], text=row["text"])
    for row in tqdm(corpus_dataset)
]
evaluator.load_corpus(documents)

# Load the QA dataset from Hugging Face
qa_dataset = load_dataset(
    "castorini/mr-tydi", "japanese", split="test", trust_remote_code=True
)
qa_pairs = [
    QandA(
        question=row["query"],
        answers=[p["docid"] for p in row["positive_passages"]],
        negative_answers=[p["docid"] for p in row["negative_passages"]],
    )
    for row in tqdm(qa_dataset)
]
evaluator.load_qa_pairs(qa_pairs)
```

### Step 3: Create and Index the Corpus

Create a new index from the loaded corpus or set an existing index:

```python
# Create a new index from the loaded corpus
# This will create an Elasticsearch index using the documents loaded into the evaluator
evaluator.create_index_from_corpus()

# Alternatively, set an existing index name to use with the evaluator
# This is useful if you already have an index and want to use it for evaluation
evaluator.set_index_name("your_existing_index_name")
```

### Step 4: Calculate Accuracy Metrics

Calculate accuracy metrics and set a custom query template for Elasticsearch search:

```python
# Optionally, set a custom query template for Elasticsearch
# {{question}} in the template will be replaced with each actual question
custom_query_template = {
    "match": {
        "text": "{{question}}"
    }
}
evaluator.set_custom_query_template(custom_query_template)

# Calculate and output the precision, recall, false positive rate, and nDCG
precision = evaluator.calculate_precision()
recall = evaluator.calculate_recall()
fpr = evaluator.calculate_fpr()
ndcg = evaluator.calculate_ndcg()

print(f"Precision: {precision}, Recall: {recall}, False Positive Rate: {fpr}, nDCG: {ndcg}")
```

## License

`elasticsearch-ir-evaluator` is available under the MIT License.
