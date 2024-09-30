# Information Retrieval

This Python module is designed as an educational and intuitive module for understanding the fundamentals of **Information Retrieval (IR)**.
It provides simple implementations of basic IR techniques with minimal dependencies.

*The module is intended for educational purposes and is not suitable for production environments.*

## Features

- **Simple Search and Ranking**: Provides simple search and ranking functionality using the Vector Space Model.
- **Automated Testing**: Includes a suite of automated tests to ensure the reliability and correctness of the implementation.
- **Minimal Dependencies**: Includes only few dependencies (word stemming & tokenise) to keep the module lightweight and easy to understand.
- **Progress Tracking**: Utilizes the `tqdm` library to provide real-time progress updates during lengthy operations, enhancing user experience.
- **Detailed Logging**: Incorporates a logging system to track the operations, helping in debugging and ensuring transparency of the process.
- **Object-Oriented Design**: The module is designed with a focus on modularity and extensibility with little abstraction (*OOP*).
  - easy to extend with new features and new weighting models
- **Optimisation Techniques**: Some optimisation techniques added to speed up the computation process.
  - cache inverse documents frequency
  - use matrix multiplication for vector operations
  - process larger documents first (the remaining documents for computation will be smaller and smaller)

## Installation

Install all the dependencies using `pip` within virtual environment:

```terminal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# install dev dependencies
pip install -r requirements-dev.txt
```

## Usage

### Data Preparation

Example of the data directory structure:

```bash
./data/
� EnglishNews
   ├── News6.txt
   ├── News78.txt
   ├── News84.txt
   ├── News2994.txt
   └── News3000.txt
```

Vector space model can be built by given collection of documents in the `EnglishNews` directory.

### Model Construction

Build vector space model in `main.py` and run:

```bash
python main.py
# or with arguments
python main.py --sample-size 1000 --query "London BBC breaking news" --logging-level CRITICAL
```

### Save and Load Model

Here's the code snippet to save and load the model, with `joblib` library:

*the `joblib` library is particularly efficient for objects that carry large numpy arrays, which might be the case with a vector space model.*

```python
import joblib

vs = VectorSpace(
    weighting_model=BM25(),
    parser=Parser(stemmer=SnowballStemmer(language="english")),
    logging_level=logging_level,
)
vs.build(documents_directory=files_path, sample_size=sample_size)

# saving the model to disk
joblib.dump(vs, os.path.join("vsm", "bm25_snwball_vs.joblib"))

# load model from disk
vs_loaded = joblib.load(os.path.join("vsm", "bm25_snwball_vs.joblib"))
vs_loaded.search("London BBC breaking news")
```

Check `main.py` for examples.

## Testing

Since `Makefile` is provided, you can run all tests with:

```bash
# run all tests
make test
# run coverage
make cov
```

## Acknowledgments

This project began as a part of a course on Web Search and Mining, taught by Professor Tsai at National Chengchi University (*NCCU*). I extend my heartfelt gratitude to Professor Tsai for his invaluable guidance and the insights that sparked the development of this module.

A special acknowledgment goes to the adage that reminds us that **software does not merely get built; it grows**.

## Advanced Module

In addition to the basic module, we have introduced an advanced module written in Golang for performance boost in heavy lifting functions. This module includes the following components:

- **Logger**: A logger setup function similar to `setup_logger` in `log.py`.
- **Metric**: A Metric struct with methods for cosine similarity and euclidean distance.
- **Model**: A Model struct with methods for weighting, vector creation, and matrix creation. It also includes TFIDF and BM25 structs that embed Model and override necessary methods.
- **Parser**: A Parser struct with methods for tokenizing, removing stopwords, and stemming words.
- **VectorSpace**: A VectorSpace struct with methods for building the model, finding related documents, searching, and ranking. It also includes a Documents struct with methods for loading, cleaning, and sorting documents.

### Installation

To use the advanced module, you need to have Golang installed. You can install Golang from [here](https://golang.org/dl/).

### Usage

To use the advanced module, you need to build the Go code and run it. Here is an example of how to do it:

```bash
cd ir/advance
go build
./advance
```

You can then use the advanced module in your Python code by importing the necessary components from the `advance` package.

## UML

TODO: UML diagram
