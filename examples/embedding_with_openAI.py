"""
This script demonstrates how to evaluate search relevance using datasets from Hugging Face.
It utilizes OpenAI's Embedding API for vectorizing text, ensuring that corpus texts are chunked to not exceed the maximum length allowed for vectorization.
The evaluation encompasses full-text search, vector search, and hybrid search methods to assess their effectiveness in retrieving relevant information.
"""
import logging
import os
import random
import re

from datasets import load_dataset
from elasticsearch import Elasticsearch
from elasticsearch_ir_evaluator.evaluator import (
    Document,
    Passage,
    ElasticsearchIrEvaluator,
    QandA,
)
from openai import OpenAI
from tqdm.auto import tqdm

logging.getLogger("httpx").setLevel(logging.CRITICAL)

MAX_TEXT_LENGTH = 8191 - 1000
OVERLAP_SIZE = 64  # Number of characters to overlap

client = OpenAI()


def vectorize_texts(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
    return [doc.embedding for doc in response.data]


MAX_TEXT_LENGTH = 8191 - 1000  # Maximum text length


def split_text(text):
    # Split the text using spaces, tabs, newlines, and periods as delimiters
    segments = re.split(r"[ \u3000\t\n。]+", text)
    chunks = []
    current_chunk = ""
    previous_segment = ""

    for segment in segments:
        if segment:  # Ignore empty segments
            if len(current_chunk + segment) + 1 <= MAX_TEXT_LENGTH - OVERLAP_SIZE:
                # If the current chunk plus the new segment is within the limit, add the segment
                current_chunk += segment + " "
            else:
                # When adding the current segment exceeds the limit
                # finalize the current chunk and start a new one
                # Ensure overlap by including the end of the current chunk in the new chunk
                if current_chunk:
                    # Avoid adding an empty chunk
                    chunks.append(current_chunk.strip())
                current_chunk = previous_segment[-OVERLAP_SIZE:] + " " + segment + " "
            previous_segment = segment  # Store the last segment added

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def compress_datasets(corpus_dataset, qa_dataset, limit):
    # Collect IDs of documents with answers and without answers
    answer_ids = {
        p["docid"]
        for qa in tqdm(qa_dataset, desc="Collecting Answer IDs")
        for p in qa["positive_passages"]
    }
    negative_ids = {p["docid"] for qa in qa_dataset for p in qa["negative_passages"]}
    all_ids = answer_ids.union(negative_ids)

    # Select corpus entries corresponding to answer and non-answer documents
    compressed_corpus = [
        doc
        for doc in tqdm(corpus_dataset, desc="Compressing Corpus")
        if doc["docid"] in all_ids
    ]

    # If the number of selected documents is less than the limit, add random documents to meet the limit
    if len(compressed_corpus) < limit:
        remaining_docs = [doc for doc in corpus_dataset if doc["docid"] not in all_ids]
        additional_docs = random.sample(remaining_docs, limit - len(compressed_corpus))
        compressed_corpus.extend(additional_docs)

    return compressed_corpus


def escape_slash(query):
    if "/" in query:
        print("escape", query, query.replace("/", "\\/"))
        return query.replace("/", "\/")
    return query


def main():

    # Load the corpus and QA datasets
    corpus_dataset = load_dataset(
        "castorini/mr-tydi-corpus", "japanese", split="train", trust_remote_code=True
    )
    qa_dataset = load_dataset(
        "castorini/mr-tydi", "japanese", split="train", trust_remote_code=True
    )

    # Compress the datasets
    compressed_corpus = compress_datasets(corpus_dataset, qa_dataset, limit=100000)

    # Initialize Elasticsearch client using environment variables
    es_client = Elasticsearch(
        hosts=os.environ["ES_HOST"],
        basic_auth=(os.environ["ES_USERNAME"], os.environ["ES_PASSWORD"]),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Create an instance of ElasticsearchIrEvaluator
    evaluator = ElasticsearchIrEvaluator(es_client)

    custom_index_settings = {
        "analysis": {
            "analyzer": {
                "ja-kuromoji": {
                    "type": "custom",
                    "char_filter": ["icu_normalizer"],
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",
                        "kuromoji_part_of_speech",
                        "ja_stop",
                        "kuromoji_number",
                        "kuromoji_stemmer",
                    ],
                }
            }
        }
    }
    text_field_config = {
        "analyzer": "ja-kuromoji",
        "search_analyzer": "ja-kuromoji",
    }
    vector_field_config = {
        "similarity": "cosine",
    }
    evaluator.set_index_settings(custom_index_settings)
    evaluator.set_text_field_config(text_field_config)
    evaluator.set_dence_vector_field_config(vector_field_config)

    documents = []
    chunk = []
    for row in tqdm(compressed_corpus):
        text = row["text"]
        if len(text) >= MAX_TEXT_LENGTH:
            split_texts = split_text(text)
            passages = []
            for t in split_texts:
                vectors = vectorize_texts([t])
                passages.append(Passage(text=t, vector=vectors[0]))
            documents.append(
                Document(id=row["docid"], title=row["title"], passages=passages)
            )
            continue

        # Add text to the current batch
        if chunk and (
            sum(len(r["text"]) for r in chunk) + len(text) >= MAX_TEXT_LENGTH
        ):
            # If adding this text exceeds MAX_TEXT_LENGTH, process the current batch first
            vectors = vectorize_texts([r["text"] for r in chunk])
            for r, vec in zip(chunk, vectors):
                documents.append(
                    Document(
                        id=r["docid"],
                        title=r["title"],
                        text=r["text"],
                        passages=[Passage(vector=vec)],
                    )
                )
            chunk = []  # Reset the batch
        chunk.append(row)

        if len(documents) >= 1000:
            evaluator.index(documents)
            documents = []
    if chunk:
        vectors = vectorize_texts([r["text"] for r in chunk])
        for r, vec in zip(chunk, vectors):
            documents.append(
                Document(
                    id=r["docid"],
                    title=r["title"],
                    text=r["text"],
                    passages=[Passage(vector=vec)],
                )
            )
        evaluator.index(documents)

    # evaluator.set_index_name("corpus_xxxxxxx_xxxxxx")

    # Load the QA dataset and vectorize each query
    qa_pairs = []
    qa_pairs = []
    batch_texts = []
    batch_indices = []
    batch_lengths = 0

    for i, row in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        query_length = len(row["query"])
        # Checks to see if the batch length has exceeded the maximum or reached the end of the data set
        if batch_lengths + query_length > MAX_TEXT_LENGTH or i == len(qa_dataset) - 1:
            vectors = vectorize_texts(batch_texts)
            for idx, vector in zip(batch_indices, vectors):
                row = qa_dataset[idx]
                q = escape_slash(row["query"])
                qa_pairs.append(
                    QandA(
                        question=q,
                        answers=[p["docid"] for p in row["positive_passages"]],
                        negative_answers=[p["docid"] for p in row["negative_passages"]],
                        vector=vector,
                    )
                )
            batch_texts = []
            batch_indices = []
            batch_lengths = 0
        if batch_lengths + query_length <= MAX_TEXT_LENGTH:
            batch_texts.append(row["query"])
            batch_indices.append(i)
            batch_lengths += query_length
    # Define a custom query template for Elasticsearch
    search_templates = [
        # Full text search
        {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": "{{question}}",
                                "fields": ["title", "text"],
                            }
                        },
                        {
                            "nested": {
                                "path": "passages",
                                "query": {
                                    "multi_match": {
                                        "query": "{{question}}",
                                        "fields": ["passages.text"],
                                    }
                                },
                            }
                        },
                    ]
                }
            }
        },
        # Vector search
        {
            "knn": [
                {
                    "field": "vector",
                    "query_vector": "{{vector}}",
                    "k": 100,
                    "num_candidates": 200,
                },
                {
                    "field": "passages.vector",
                    "query_vector": "{{vector}}",
                    "k": 100,
                    "num_candidates": 200,
                },
            ],
        },
        # Hybrid search
        {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": "{{question}}",
                                "fields": ["title", "text"],
                            }
                        },
                        {
                            "nested": {
                                "path": "passages",
                                "query": {
                                    "multi_match": {
                                        "query": "{{question}}",
                                        "fields": ["passages.text"],
                                    }
                                },
                            }
                        },
                    ]
                }
            },
            "knn": [
                {
                    "field": "vector",
                    "query_vector": "{{vector}}",
                    "k": 100,
                    "num_candidates": 200,
                },
                {
                    "field": "passages.vector",
                    "query_vector": "{{vector}}",
                    "k": 100,
                    "num_candidates": 200,
                },
            ],
            "rank": {
                "rrf": {
                    "window_size": 100,
                }
            },
        },
    ]
    results = []

    for query in search_templates:
        evaluator.set_search_template(query)
        result = evaluator.calculate(qa_pairs, top_n=100)
        results.append(result)

    for r in results:
        print(r.to_markdown())


if __name__ == "__main__":
    main()
