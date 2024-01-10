import json
import logging
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import BulkIndexError, bulk

from .types import Document, Passage, QandA, Result


class ElasticsearchIrEvaluator:
    def __init__(self, es_client: Elasticsearch):
        """
        Constructor for the ElasticsearchIrEvaluator class.

        This class provides tools for evaluating Information Retrieval (IR) systems using Elasticsearch.
        The constructor initializes the Elasticsearch client and sets up initial configurations.

        Args:
            es_client (Elasticsearch): An instance of the Elasticsearch client.
        """
        self.es = es_client
        self.max_bulk_size = 5000
        self.max_retries = 3
        self.top_n = 100
        self.index_name = None
        self.index_settings = None
        self.text_field_config = None
        self.search_template = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("elastic_transport.transport").setLevel(logging.CRITICAL)

    def set_log_level(self, level: int) -> None:
        """
        Set the logging level for this evaluator.

        This method allows the user to set the logging level to control the amount and type of logs generated.

        Args:
            level (int): The logging level to set. Example: logging.INFO, logging.DEBUG.
        """
        self.logger.setLevel(level)

    def set_index_name(self, index_name: str):
        """
        Sets the name for the Elasticsearch index.

        This method is used to set the name of the Elasticsearch index to be used.

        Args:
            index_name (str): The name of the index to be set.
        """
        self.index_name = index_name
        self.logger.info(f"Index name set to: {self.index_name}")

    def set_index_settings(self, index_settings: Dict):
        """
        Set custom settings for the Elasticsearch index.

        This method is used to provide custom settings for the Elasticsearch index, such as the number of shards.

        Args:
            index_settings (Dict): A dictionary containing index settings.
        """
        self.index_settings = index_settings

    def set_text_field_config(self, text_field_config: Dict):
        """
        Set configuration for the text fields of the Elasticsearch index.

        This method is used to set specific configurations for text fields, like analyzers or term vectors.

        Args:
            text_field_config (Dict): A dictionary containing text field configurations.
        """
        self.text_field_config = text_field_config

    def _create_index(self, sample_document: Document) -> None:
        """
        Creates an Elasticsearch index based on a sample document.

        This private method automatically generates an index with appropriate settings and mappings
        based on the provided sample document, including handling vector fields if present.

        Args:
            sample_document (Document): A sample document used to determine the structure of the index.
        """
        self.index_name = f'corpus_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        text_field_settings = {"type": "text"}
        # Ensure "type": "text" is always included in text_field_config
        if self.text_field_config is not None:
            text_field_settings = text_field_settings | {**self.text_field_config}

        mapping = {
            "properties": {
                "id": {"type": "keyword"},
                "title": text_field_settings,
                "text": text_field_settings,
                "passages": {
                    "type": "nested",
                    "properties": {"text": text_field_settings},
                },
            }
        }
        # Check if any Document in the corpus has a vector and set the dims accordingly
        vector_dims = None
        if sample_document.vector:
            vector_dims = len(sample_document.vector)
        # Check for vector dimensions in passages
        if sample_document.passages:
            for passage in sample_document.passages:
                if passage.vector:
                    vector_dims = len(passage.vector)
                    break

        # Add the vector field to the mapping if vector_dims is found
        if vector_dims:
            mapping["properties"]["vector"] = {
                "type": "dense_vector",
                "dims": vector_dims,
                "index": True,
                "similarity": "cosine",
            }
            mapping["properties"]["passages"]["properties"]["vector"] = {
                "type": "dense_vector",
                "dims": vector_dims,
            }

        # Index settings
        index = {
            "mappings": mapping,
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        }

        # Update index settings if provided
        if self.index_settings:
            index["settings"] = self.index_settings

        self.es.indices.create(index=self.index_name, body=index)
        self.logger.info(
            f"Index {self.index_name} created with settings: \n{json.dumps(index, indent=2)}"
        )

    def _bulk(self, actions: List[Dict]):
        """
        Executes a bulk indexing operation in Elasticsearch.

        This private method attempts to perform a bulk indexing operation, and retries if it encounters
        a BulkIndexError, up to a maximum number of retries.

        Args:
            actions (List[Dict]): A list of actions (documents) to be bulk indexed.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                bulk(self.es, actions)
                break
            except BulkIndexError as e:
                self.logger.warning(f"Bulk indexing failed: {e.errors}")
                actions = [
                    action
                    for action, error in zip(actions, e.errors)
                    if error is not None
                ]
                retries += 1
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                break

    def index(self, documents: List[Document], pipeline: Optional[str] = None) -> None:
        """
        Index the given documents in Elasticsearch.

        This method takes a list of documents and indexes them in Elasticsearch. It handles bulk indexing and monitors progress.
        An optional ingest pipeline can be specified for preprocessing the documents before indexing.

        Args:
            documents (List[Document]): A list of documents to be indexed. Each document should be an instance of the Document class.
            pipeline (str, optional): The name of the ingest pipeline to be used for preprocessing documents.
        """

        if not self.index_name:
            self._create_index(documents[0])

        actions = []
        doc_count = len(documents)
        end = 0
        self.logger.info(f"Indexing {doc_count} documents...")
        for doc in documents:
            action = {"_index": self.index_name, "_source": doc.dict()}
            if pipeline:
                action["_op_type"] = "index"
                action["pipeline"] = pipeline
            actions.append(action)

            if len(actions) >= self.max_bulk_size:
                self._bulk(actions)
                end += len(actions)
                actions = []

                progress = (end / doc_count) * 100
                sys.stdout.write(
                    f"\rIndexed {end} / {doc_count} documents ({progress:.2f}%)"
                )
                sys.stdout.flush()

        # イテレータの最後の部分をインデックス
        if actions:
            self._bulk(actions)
            end += len(actions)
            progress = (end / doc_count) * 100
            sys.stdout.write(
                f"\rIndexed {end} / {doc_count} documents ({progress:.2f}%)"
            )
            sys.stdout.flush()

        print()
        self.logger.info("Indexing completed.")

    def set_search_template(self, search_template: Dict):
        """
        Set a custom search template for Elasticsearch queries.

        This method allows setting up a custom query template for more advanced or specific search requirements.
        In the provided QA dataset iterator, each question is inserted into the '{{question}}' placeholder,
        and if present, the question's vector is inserted into the '{{vector}}' placeholder in the template.

        Args:
            search_template (Dict): A dictionary representing the search template.
        """
        self.search_template = search_template

    def _replace_template(self, template: Union[Dict, List, str], qa_pair: QandA):
        """
        Replace placeholders in the search template with actual query data.

        This private method replaces the placeholders '{{question}}' and '{{vector}}' in the search template
        with the actual question and vector from the QandA object. The '{{question}}' placeholder is replaced
        with the text of the question, and '{{vector}}', if present, is replaced with the question's vector.

        Args:
            template (Union[Dict, List, str]): The search template with placeholders.
            qa_pair (QandA): The question-answer pair containing the question and optional vector.
        """
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
        """
        Perform a search query in Elasticsearch using a given question-answer pair.

        This private method executes a search query in Elasticsearch based on the question in the given QandA object.
        It uses the custom search template if set, or a default match query otherwise.

        Args:
            qa_pair (QandA): The question-answer pair to base the search query on.

        Returns:
            List[str]: A list of document IDs that match the search query.
        """
        if self.search_template is None:
            search_body = {"query": {"match": {"text": qa_pair.question}}}
        else:
            search_body = self._replace_template(self.search_template, qa_pair)

        search_body["_source"] = ["id"]

        response = self.es.search(
            index=self.index_name, body=search_body, size=self.top_n
        )
        return [hit["_source"]["id"] for hit in response["hits"]["hits"]]

    def calculate_precision(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the precision of the search results.

        This method computes the precision metric for the search results given a list of question-answer pairs.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.
        """
        if top_n is not None:
            self.top_n = top_n
        total_precision = 0

        for qa_pair in qa_pairs:
            correct_answers = set(qa_pair.answers)

            search_results = set(self._search(qa_pair))
            relevant_retrieved = len(search_results & correct_answers)
            total_retrieved = len(search_results)

            precision = (
                relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
            )
            total_precision += precision

        return total_precision / len(qa_pairs) if qa_pairs else 0

    def calculate_recall(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the recall of the search results.

        This method computes the recall metric for the search results given a list of question-answer pairs,
        which is the proportion of relevant documents that are successfully retrieved.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.
        """
        if top_n is not None:
            self.top_n = top_n
        total_recall = 0

        for qa_pair in qa_pairs:
            correct_answers = set(qa_pair.answers)

            search_results = set(self._search(qa_pair))
            relevant_retrieved = len(search_results & correct_answers)
            total_relevant = len(correct_answers)

            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            total_recall += recall

        return total_recall / len(qa_pairs) if qa_pairs else 0

    def calculate_mrr(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the Mean Reciprocal Rank (MRR) of the search results.

        This method calculates the MRR metric, which is the average of reciprocal ranks of the first relevant answer
        for a set of queries.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.
        """
        if top_n is not None:
            self.top_n = top_n

        total_mrr = 0

        for qa_pair in qa_pairs:
            correct_answers = set(qa_pair.answers)
            search_results = self._search(qa_pair)

            for rank, result in enumerate(search_results, 1):
                if result in correct_answers:
                    total_mrr += 1 / rank
                    break

        return total_mrr / len(qa_pairs) if qa_pairs else 0

    def calculate_fpr(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the False Positive Rate (FPR) of the search results.

        This method calculates the FPR metric, which is the ratio of the number of false positives to the total
        number of actual negatives.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation, including negative answers.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.
        """
        if top_n is not None:
            self.top_n = top_n
        total_fpr = 0
        num_pairs = len(qa_pairs)

        for qa_pair in qa_pairs:
            incorrect_answers = set(qa_pair.negative_answers)

            search_results = set(self._search(qa_pair))
            false_positives = len(search_results & incorrect_answers)
            true_negatives = len(incorrect_answers - search_results)

            if (false_positives + true_negatives) > 0:
                total_fpr += false_positives / (false_positives + true_negatives)

        return total_fpr / num_pairs if num_pairs > 0 else 0

    def calculate_ndcg(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the normalized Discounted Cumulative Gain (nDCG) of the search results.

        This method calculates the nDCG metric, a measure of ranking quality that considers the position
        of correct answers, penalizing correct answers that appear lower in the search results.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.
        """
        if top_n is not None:
            self.top_n = top_n

        def dcg(scores):
            return np.sum([(2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores)])

        total_ndcg = 0

        for qa_pair in qa_pairs:
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

        return total_ndcg / len(qa_pairs) if qa_pairs else 0

    def calculate_map(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the Mean Average Precision (MAP).

        MAP is the mean of the Average Precision scores for each query, which is the average of the precision scores
        after each relevant document is retrieved.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.

        Returns:
            float: The MAP score.
        """
        if top_n is not None:
            self.top_n = top_n
        total_average_precision = 0

        for qa_pair in qa_pairs:
            search_results = set(self._search(qa_pair))
            relevant_documents = set(qa_pair.answers)
            average_precision, relevant_count = 0, 0

            for i, doc_id in enumerate(search_results, 1):
                if doc_id in relevant_documents:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    average_precision += precision_at_i

            total_average_precision += (
                average_precision / len(relevant_documents) if relevant_documents else 0
            )

        return total_average_precision / len(qa_pairs) if qa_pairs else 0

    def calculate_cg(self, qa_pairs: List[QandA], top_n: Optional[int] = None) -> float:
        """
        Calculate the Cumulative Gain (CG).

        CG is the sum of the relevancy scores of the top N search results. The relevancy score is binary (relevant or not).

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.

        Returns:
            float: The CG score.
        """
        if top_n is not None:
            self.top_n = top_n
        total_cg = 0

        for qa_pair in qa_pairs:
            search_results = self._search(qa_pair)
            relevant_documents = set(qa_pair.answers)
            cg_score = sum(
                1 for doc_id in search_results if doc_id in relevant_documents
            )

            total_cg += cg_score

        return total_cg / len(qa_pairs) if qa_pairs else 0

    def calculate_bpref(
        self, qa_pairs: List[QandA], top_n: Optional[int] = None
    ) -> float:
        """
        Calculate the Binary Preference (BPref).

        BPref measures how often relevant documents are ranked above non-relevant documents. It is based on pairs of
        relevant and non-relevant documents.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.

        Returns:
            float: The BPref score.
        """
        if top_n is not None:
            self.top_n = top_n
        total_bpref = 0

        for qa_pair in qa_pairs:
            search_results = self._search(qa_pair)
            relevant_documents = set(qa_pair.answers)
            non_relevant_documents = (
                set(qa_pair.negative_answers) if qa_pair.negative_answers else set()
            )

            bpref_score = 0
            for doc_id in search_results:
                if doc_id in relevant_documents:
                    bpref_score += 1 - (
                        len(
                            [
                                d
                                for d in non_relevant_documents
                                if d in search_results
                                and search_results.index(d)
                                > search_results.index(doc_id)
                            ]
                        )
                        / len(non_relevant_documents)
                    )

            total_bpref += (
                bpref_score / len(relevant_documents) if relevant_documents else 0
            )

        return total_bpref / len(qa_pairs) if qa_pairs else 0

    def calculate(self, qa_pairs: List[QandA], top_n: Optional[int] = None) -> Result:
        """
        Calculate all possible metrics based on the given QA pairs in a single iteration.

        This method calculates various search evaluation metrics such as Precision, Recall, FPR, nDCG, MAP, CG, BPref, and MRR
        based on the available data in the QA pairs. It does so by performing a single search per QA pair and reusing the results for each metric.

        Args:
            qa_pairs (List[QandA]): A list of question-answer pairs for evaluation.
            top_n (int, optional): The number of top results to consider for evaluation. Defaults to the class's top_n attribute.

        Returns:
            Result: An instance of MetricsResults containing the calculated metrics.
        """
        if top_n is not None:
            self.top_n = top_n

        sum_precision = (
            sum_recall
        ) = sum_fpr = sum_ndcg = sum_map = sum_cg = sum_bpref = sum_mrr = 0
        total_pairs = len(qa_pairs)

        for qa_pair in qa_pairs:
            search_results = self._search(qa_pair)
            relevant_documents = set(qa_pair.answers)
            non_relevant_documents = (
                set(qa_pair.negative_answers) if qa_pair.negative_answers else set()
            )

            # MAP, BPref, and MRR variables
            average_precision = bpref_score = mrr_added = 0
            relevant_count = 0

            # nDCG variables
            dcg = idcg = 0

            # FPR variables
            false_positives = len(set(search_results) & non_relevant_documents)
            true_negatives = len(non_relevant_documents - set(search_results))

            for i, doc_id in enumerate(search_results, 1):
                if doc_id in relevant_documents:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    average_precision += precision_at_i
                    dcg += 1 / np.log2(i + 1)

                    if not mrr_added:
                        sum_mrr += 1 / i
                        mrr_added = True

                    non_relevant_lower_rank = len(
                        [
                            d
                            for d in non_relevant_documents
                            if d in search_results
                            and search_results.index(d) > search_results.index(doc_id)
                        ]
                    )
                    bpref_score += (
                        1 - (non_relevant_lower_rank / len(non_relevant_documents))
                        if non_relevant_documents
                        else 1
                    )

                if i <= len(relevant_documents):
                    idcg += 1 / np.log2(i + 1)

            # Update cumulative sums
            sum_map += (
                average_precision / len(relevant_documents) if relevant_documents else 0
            )
            sum_bpref += (
                bpref_score / len(relevant_documents) if relevant_documents else 0
            )
            sum_ndcg += dcg / idcg if idcg > 0 else 0
            sum_cg += relevant_count

            # Precision and Recall
            true_positives = relevant_count
            false_positives = len(search_results) - true_positives
            sum_precision += (
                true_positives / len(search_results) if search_results else 0
            )
            sum_recall += (
                true_positives / len(relevant_documents) if relevant_documents else 0
            )

            sum_fpr += (
                false_positives / (false_positives + true_negatives)
                if (false_positives + true_negatives) > 0
                else 0
            )

        results = Result(
            Precision=sum_precision / total_pairs,
            Recall=sum_recall / total_pairs,
            FPR=sum_fpr / total_pairs if non_relevant_documents else None,
            nDCG=sum_ndcg / total_pairs,
            MAP=sum_map / total_pairs,
            CG=sum_cg / total_pairs,
            BPref=sum_bpref / total_pairs if non_relevant_documents else None,
            MRR=sum_mrr / total_pairs,
        )

        return results
