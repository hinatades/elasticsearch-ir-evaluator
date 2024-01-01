import sys
from datetime import datetime
from typing import Callable, Dict, List, Union

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, bulk

from .types import Document, QandA


class ElasticsearchIrEvaluator:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.corpus = []
        self.qa_pairs = []
        self.index_name = None
        self.top_n = 100
        self.search_template = None

    def load_corpus(self, corpus: List[Document]) -> None:
        """Load the corpus."""
        self.corpus = corpus

    def load_qa_pairs(self, qa_pairs: List[QandA]) -> None:
        self.qa_pairs = qa_pairs

    def set_index_name(self, index_name: str):
        """Set the name for the Elasticsearch index."""
        self.index_name = index_name
        print(f"Index name set to: {self.index_name}")

    def create_index_from_corpus(self, index_settings=None) -> None:
        """Create an index in Elasticsearch using the loaded corpus.

        Args:
            index_settings: Optional. Custom settings for the Elasticsearch index.
        """
        self.index_name = f'corpus_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        mapping = {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        }
        # Check if any Document in the corpus has a vector and set the dims accordingly
        vector_dims = None
        for doc in self.corpus:
            if doc.vector:
                vector_dims = len(doc.vector)
                break

        # Add the vector field to the mapping if vector_dims is found
        if vector_dims:
            mapping["properties"]["vector"] = {
                "type": "dense_vector",
                "dims": vector_dims,
                "index": True,
                "similarity": "cosine",
            }

        # Index settings
        index = {
            "mappings": mapping,
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        }

        # Update index settings if provided
        if index_settings:
            index["settings"] = index_settings

        self.es.indices.create(index=self.index_name, body=index)
        print(f"Index {self.index_name} created with settings: {index}")

    def index_corpus(
        self, document_transformer: Callable[[Document], Document] = None, max_retries=3
    ) -> None:
        """Index the corpus in Elasticsearch. An optional transformer can be applied to each documents."""
        if document_transformer:
            self.corpus = [document_transformer(doc) for doc in self.corpus]

        doc_count = len(self.corpus)
        print(f"Indexing {doc_count} documents...")

        bulk_size = 5000
        for i in range(0, doc_count, bulk_size):
            end = min(i + bulk_size, doc_count)
            actions = [
                {"_index": self.index_name, "_source": self.corpus[j].dict()}
                for j in range(i, end)
            ]

            retries = 0
            while retries < max_retries:
                try:
                    bulk(self.es, actions)
                    break
                except BulkIndexError as e:
                    print(f"Bulk indexing failed: {e.errors}")
                    actions = [
                        action
                        for action, error in zip(actions, e.errors)
                        if error is not None
                    ]
                    retries += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

            progress = (end / doc_count) * 100
            sys.stdout.write(
                f"\rIndexed {end} / {doc_count} documents ({progress:.2f}%)"
            )
            sys.stdout.flush()

        print("\nIndexing completed.")

    def set_search_template(self, search_template: Dict):
        """Set a custom search template for Elasticsearch queries."""
        self.search_template = search_template

    def _replace_template(self, template: Union[Dict, List, str], qa_pair: QandA):
        if isinstance(template, dict):
            return {k: self._replace_template(v, qa_pair) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._replace_template(item, qa_pair) for item in template]
        elif isinstance(template, str):
            template = template.replace("{{question}}", qa_pair.question)
            if qa_pair.vector is not None and "{{vector}}" in template:
                return qa_pair.vector
            return template
        else:
            return template

    def _search(self, qa_pair: QandA) -> List[str]:
        if self.search_template is None:
            search_body = {"query": {"match": {"text": qa_pair.question}}}
        else:
            search_body = self._replace_template(self.search_template, qa_pair)

        search_body["_source"] = ["id"]

        response = self.es.search(
            index=self.index_name, body=search_body, size=self.top_n
        )
        return [hit["_source"]["id"] for hit in response["hits"]["hits"]]

    def calculate_precision(self, top_n: int = None) -> float:
        """Calculate the precision of the search results."""
        self.top_n = top_n if top_n is not None else self.top_n
        total_precision = 0

        for qa_pair in self.qa_pairs:
            correct_answers = set(qa_pair.answers)

            search_results = set(self._search(qa_pair))
            relevant_retrieved = len(search_results & correct_answers)
            total_retrieved = len(search_results)

            precision = (
                relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
            )
            total_precision += precision

        return total_precision / len(self.qa_pairs) if self.qa_pairs else 0

    def calculate_recall(self, top_n: int = None) -> float:
        """Calculate the recall of the search results."""
        self.top_n = top_n if top_n is not None else self.top_n
        total_recall = 0

        for qa_pair in self.qa_pairs:
            correct_answers = set(qa_pair.answers)

            search_results = set(self._search(qa_pair))
            relevant_retrieved = len(search_results & correct_answers)
            total_relevant = len(correct_answers)

            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            total_recall += recall

        return total_recall / len(self.qa_pairs) if self.qa_pairs else 0

    def calculate_mrr(self, top_n: int = None) -> float:
        """Calculate the Mean Reciprocal Rank (MRR) of the search results."""
        self.top_n = top_n if top_n is not None else self.top_n
        total_mrr = 0

        for qa_pair in self.qa_pairs:
            correct_answers = set(qa_pair.answers)
            search_results = self._search(qa_pair)

            for rank, result in enumerate(search_results, 1):
                if result in correct_answers:
                    total_mrr += 1 / rank
                    break

        return total_mrr / len(self.qa_pairs) if self.qa_pairs else 0

    def calculate_fpr(self) -> float:
        """Calculate the False Positive Rate (FPR) of the search results."""
        false_positives = 0
        true_negatives = 0

        for qa_pair in self.qa_pairs:
            incorrect_answers = set(qa_pair.negative_answers)

            search_results = set(self._search(qa_pair))
            false_positives += len(search_results & incorrect_answers)
            true_negatives += len(incorrect_answers - search_results)

        return (
            false_positives / (false_positives + true_negatives)
            if (false_positives + true_negatives) > 0
            else 0
        )

    def calculate_ndcg(self) -> float:
        """Calculate the normalized Discounted Cumulative Gain (nDCG) of the search results."""

        def dcg(scores):
            return np.sum([(2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores)])

        total_ndcg = 0

        for qa_pair in self.qa_pairs:
            correct_answers = set(qa_pair.answers)

            search_results = self._search(qa_pair)
            relevance_scores = [
                1 if result in correct_answers else 0 for result in search_results
            ]
            ideal_scores = sorted(relevance_scores, reverse=True)

            DCG = dcg(relevance_scores)
            IDCG = dcg(ideal_scores)

            nDCG = DCG / IDCG if IDCG > 0 else 0
            total_ndcg += nDCG

        return total_ndcg / len(self.qa_pairs) if self.qa_pairs else 0
