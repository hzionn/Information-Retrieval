"""
a custom parser that tokenises a document, removes stopwords, and stems words.
"""

import os
import string
from typing import List

from .log import setup_logger


class Parser:
    def __init__(self, stemmer=None, language: str = "english", logging_level="INFO"):
        self._logger = setup_logger(
            filename=__file__,
            classname=self.__class__.__name__,
            level=logging_level.upper(),
        )
        self._language = language
        self._stemmer = stemmer
        self.punctuations = self._get_punctuation()
        self.stopwords = self._get_stopwords()

    def _get_punctuation(self) -> List[str]:
        punctuations = list(string.punctuation)
        more_punctuations = ["（", "）", "，", "“", "”"]
        return punctuations + more_punctuations

    def _get_stopwords(self) -> List[str]:
        stopwords_file_path = self._get_stopwords_file_path()
        with open(stopwords_file_path, "r") as file:
            stopwords = [word.strip() for word in file.readlines()]
        more_stopwords = ["I"]
        return stopwords + more_stopwords

    def _get_stopwords_file_path(self) -> str:
        stopwords_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stopwords")
        stopwords_file = "EnglishStopwords.txt"
        return os.path.join(stopwords_path, stopwords_file)

    def tokenise(self, document_content: str) -> List[str]:
        """
        tokenise a document content and stem words for English.

        Args:
            document_content (str): the content of a single document

        Return:
            List[str]: a list of word tokens
        """
        # TODO: clean punctuation
        return [self.stem(word.strip()) for word in document_content.split()]

    def remove_stopwords(self, words_list: List[str]) -> List[str]:
        return [word for word in words_list if word not in self.stopwords]

    def _clean_punctuation(self, string: str) -> str:
        return string.translate(str.maketrans("".join(self.punctuations), ' '*len(self.punctuations)))

    def stem(self, word: str) -> str:
        if not self._stemmer:
            raise ValueError("Stemmer is not defined.") 
        return self._stemmer.stem(word)


if __name__ == "__main__":
    pass
