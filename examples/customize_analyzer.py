import os

from datasets import load_dataset
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

from elasticsearch_ir_evaluator.evaluator import (Document,
                                                  ElasticsearchIrEvaluator,
                                                  QandA)


def main():
    # Initialize Elasticsearch client using environment variables
    es_client = Elasticsearch(
        hosts=os.environ["ES_HOST"],
        basic_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Create an instance of ElasticsearchIrEvaluator
    evaluator = ElasticsearchIrEvaluator(es_client)

    # Load the corpus dataset and vectorize the text of each document
    corpus_dataset = load_dataset(
        "castorini/mr-tydi-corpus",
        "japanese",
        split="train",
        trust_remote_code=True,
    )

    custom_index_settings = {
        "analysis": {
            "filter": {
                "katakana_stemmer": {
                    "type": "kuromoji_stemmer",
                    "minimum_length": "5",
                },
            },
            "char_filter": {
                "normalize": {
                    "mode": "compose",
                    "name": "nfkc_cf",
                    "type": "icu_normalizer",
                }
            },
            "tokenizer": {
                "index-kuromoji-tokenizer": {
                    "mode": "normal",
                    "type": "kuromoji_tokenizer",
                },
                "search-kuromoji-tokenizer": {
                    "mode": "search",
                    "discard_compound_token": True,
                    "type": "kuromoji_tokenizer",
                },
            },
            "analyzer": {
                "ja-index-kuromoji": {
                    "type": "custom",
                    "tokenizer": "index-kuromoji-tokenizer",
                    "filter": [
                        "katakana_stemmer",
                    ],
                    "char_filter": ["normalize"],
                },
                "ja-search-kuromoji": {
                    "type": "custom",
                    "tokenizer": "search-kuromoji-tokenizer",
                    "filter": [],
                    "char_filter": [],
                },
            },
        }
    }
    text_field_config = {
        "analyzer": "ja-index-kuromoji",
        "search_analyzer": "ja-search-kuromoji",
    }

    documents = []
    for i, row in enumerate(tqdm(corpus_dataset)):
        documents.append(
            Document(
                id=row["docid"],
                title=row["title"],
                text=row["text"],
            )
        )
    evaluator.load_corpus(documents)
    evaluator.index_corpus()
    evaluator.create_index_from_corpus(
        index_settings=custom_index_settings, text_field_config=text_field_config
    )

    # Load the QA dataset and vectorize each query
    qa_dataset = load_dataset(
        "castorini/mr-tydi",
        "japanese",
        split="test",
        trust_remote_code=True,
    )
    qa_pairs = []
    for i, row in enumerate(tqdm(qa_dataset)):
        qa_pairs.append(
            QandA(
                question=row["query"],
                answers=[p["docid"] for p in row["positive_passages"]],
                negative_answers=[p["docid"] for p in row["negative_passages"]],
            )
        )

    evaluator.load_qa_pairs(qa_pairs)

    # Define a custom query template for Elasticsearch
    search_template = {
        "query": {
            "bool": {
                "must": {"match": {"text": "{{question}}"}},
            }
        },
    }
    evaluator.set_search_template(search_template)

    # Calculate and print the Mean Reciprocal Rank (MRR)
    mrr = evaluator.calculate_mrr()
    print(f"MRR: {mrr}")


if __name__ == "__main__":
    main()
