# http://tartarus.org/~martin/PorterStemmer/python.txt
import os
from sys import maxunicode
from unicodedata import category

from nltk.stem.porter import PorterStemmer


class Parser:

    # A processor for removing the commoner morphological and inflexional endings from words in English

    def __init__(self):
        self.stemmer = PorterStemmer()
        _punctuation = dict.fromkeys(
            i for i in range(maxunicode) if category(chr(i)).startswith("P")
        )
        self.punctuation = {_punctuation[key]: " " for key in _punctuation.keys()}
        self.stopwords = [
            w.strip()
            for w in open(
                os.path.join(
                    os.path.dirname(os.getcwd()), "stopwords", "EnglishStopwords.txt"
                ),
                "r",
            ).readlines()
        ]

    def clean(self, string: str) -> str:
        """remove any nasty grammar tokens from string"""
        string = string.replace(".", " ")
        string = string.replace("\s+", " ")
        string = string.replace('"', " ")
        string = string.replace("'", " ")
        string = string.replace("'", " ")
        string = string.replace("“", " ")
        string = string.replace("”", " ")
        string = string.replace("’", " ")
        string = string.lower()
        return string

    def removeStopWords(self, list: list) -> list[str]:
        """Remove common words which have no search value"""
        return [word for word in list if word not in self.stopwords]

    def tokenise(self, string: str) -> list[str]:
        """break string up into tokens and stem words"""
        string = string.translate(self.punctuation)
        string = self.clean(string)
        # words = string.split(" ")
        words = string.split()

        return [self.stemmer.stem(word.strip()) for word in words]
