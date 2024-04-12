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
- **Object-Oriented Design**: The module is designed with a focus on modularity and extensibility with little abstraction (OOP).
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
```

## Usage

Build vector space model in `main.py` and run:

```bash
python main.py
# or with arguments
python main.py --sample-size 1000 --query "London BBC breaking news" --logging-level CRITICAL
```

Check `main.py` for examples.

## Testing

Since `Makefile` is provided, you can run all tests with:

```bash
# install dev dependencies
pip install -r requirements-dev.txt
# run all tests
make test
# run coverage
make cov
```

## Acknowledgments

This project began as a part of a course on Web Search and Mining, taught by Professor Tsai at National Chengchi University (NCCU). I extend my heartfelt gratitude to Professor Tsai for his invaluable guidance and the insights that sparked the development of this module.

A special acknowledgment goes to the adage that reminds us that **software does not merely get built; it grows**.

## UML

TODO: UML diagram
