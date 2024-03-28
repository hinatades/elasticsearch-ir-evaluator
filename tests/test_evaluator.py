import json
import os
from tempfile import TemporaryDirectory
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


def test_log_file_creation_on_sigterm():
    with TemporaryDirectory() as temp_dir:
        # Create a dummy Elasticsearch client for testing
        es_client_mock = Mock()

        # Initialize the ElasticsearchIrEvaluator instance within the temporary directory
        evaluator = ElasticsearchIrEvaluator(es_client_mock)
        evaluator.log_file_path = os.path.join(
            temp_dir, "elasticsearch-ir-evaluator-log.json"
        )

        # Directly invoke the sigterm_handler to simulate the creation of the log file upon SIGTERM signal
        with pytest.raises(SystemExit) as exit_exception:
            evaluator.sigterm_handler(None, None)

        assert exit_exception.value.code == 0

        # Check the existence of the log file
        assert os.path.exists(evaluator.log_file_path)

        # Validate the contents of the log file
        with open(evaluator.log_file_path, "r") as log_file:
            log_data = json.load(log_file)
            assert "last_processed_id" in log_data
            assert "index_name" in log_data
            assert "processed_count" in log_data
            assert "last_checkpoint_timestamp" in log_data
