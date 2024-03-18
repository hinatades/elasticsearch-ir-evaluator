from setuptools import find_packages, setup

setup(
    name="elasticsearch_ir_evaluator",
    version="0.3.1",
    description="A Python package for easily calculating information retrieval (IR) accuracy metrics using Elasticsearch and datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    url="https://github.com/hinatades/elasticsearch-ir-evaluator",
    author="Taisuke Hinata",
    author_email="hnttisk@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "elasticsearch>=8.11",
        "numpy",
        "pydantic",
        "tqdm",
    ],
)
