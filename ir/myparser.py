"""
a custom parser that tokenises a document, removes stopwords, and stems words.
(works only for English and Chinese languages)
"""

import os
import string
from typing import List

import jieba
from nltk.stem.porter import PorterStemmer


class Parser:
    def __init__(self, stemmer=PorterStemmer(), language: str = "english"):
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
        stopwords = [w.strip() for w in open(stopwords_file_path, "r").readlines()]
        more_stopwords = ["I"]
        return stopwords + more_stopwords

    def _get_stopwords_file_path(self) -> str:
        if self._language not in ("english", "chinese"):
            raise ValueError("Language must be either 'english' or 'chinese'.")
        stopwords_path = os.path.join(os.path.dirname(__file__), "stopwords")
        stopwords_file = (
            "EnglishStopwords.txt"
            if self._language == "english"
            else "ChineseStopwords.txt"
        )
        return os.path.join(stopwords_path, stopwords_file)

    def tokenise(self, document_content: str) -> List[str]:
        """
        tokenise a document content and stem words for English.

        Args:
            document_content(str): the content of a document

        Return:
            List[str]: a list of word tokens
        """
        # TODO: clean punctuation
        if self._language == "english":
            return [self.stem(word.strip()) for word in document_content.split()]
        else:
            return jieba.lcut_for_search(document_content)

    def remove_stopwords(self, words_list: List[str]) -> List[str]:
        return [word for word in words_list if word not in self.stopwords]

    def _clean_punctuation(self, string: str) -> str:
        # TODO: replace punctuation with space
        return string.translate(str.maketrans("", "", "".join(self.punctuations)))

    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)


if __name__ == "__main__":
    parser = Parser()
    # parser = Parser(language="french")
    # print(parser.punctuations)
    # print(parser.stopwords)
    print(parser.stem("running"))
    print(parser._clean_punctuation("running."))
    print(parser._clean_punctuation("we're family."))
