"""Setup configuration for the grag-simulation package."""

from setuptools import setup, find_packages
import os

# Read the long description from README.md if it exists
here = os.path.abspath(os.path.dirname(__file__))
long_description = ""
readme_path = os.path.join(here, "README.md")
if os.path.isfile(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements from requirements.txt
requirements = []
req_path = os.path.join(here, "requirements.txt")
if os.path.isfile(req_path):
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="grag-simulation",
    version="1.0.0",
    description=(
        "A pure-Python simulation of Graph Retrieval-Augmented Generation (Graph RAG) "
        "algorithms including Microsoft GraphRAG, RAPTOR, HippoRAG, and hybrid search."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Graph RAG Simulation Contributors",
    author_email="vietcuong2k182@gmail.com",
    url="https://github.com/example/grag-simulation",
    project_urls={
        "Bug Tracker": "https://github.com/example/grag-simulation/issues",
        "Documentation": "https://github.com/example/grag-simulation#readme",
        "Source Code": "https://github.com/example/grag-simulation",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "graph-rag",
        "retrieval-augmented-generation",
        "knowledge-graph",
        "graphrag",
        "community-detection",
        "pagerank",
        "raptor",
        "hipporag",
        "nlp",
        "information-retrieval",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "flake8>=6.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "ipywidgets>=8.0",
        ],
        "viz": [
            "matplotlib>=3.6",
            "seaborn>=0.12",
        ],
    },
    entry_points={
        "console_scripts": [
            # Example: run a quick benchmark from the command line
            # "grag-bench=grag.evaluation.benchmark:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
