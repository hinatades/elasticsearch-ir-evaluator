from unittest.mock import Mock

import pytest

# Import the ElasticsearchIrEvaluator and QandA from the local module
from elasticsearch_ir_evaluator import ElasticsearchIrEvaluator, QandA


def test_calculate_mrr():
    # Create a mock for the Elasticsearch client
    es_client_mock = Mock()
    # Mock the search method to return predefined results
    es_client_mock.search.return_value = {
        "hits": {"hits": [{"_source": {"id": "doc1"}}, {"_source": {"id": "doc2"}}]}
    }

    # Create an instance of ElasticsearchIrEvaluator with the mock client
    evaluator = ElasticsearchIrEvaluator(es_client=es_client_mock)

    # Load mock QA pairs
    qa_pairs = [
        QandA(question="test query", answers=["doc1"], negative_answers=["doc3"])
    ]

    # Calculate MRR
    mrr = evaluator.calculate_mrr(qa_pairs)

    # Assert the expected MRR value
    assert mrr == 1.0  # Adjust the expected value of MRR as needed
