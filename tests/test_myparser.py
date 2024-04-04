import os

from nltk.stem import PorterStemmer
import pytest

from ir.myparser import Parser


@pytest.fixture
def parser_default():
    return Parser()


@pytest.fixture
def parser_porter():
    return Parser(stemmer=PorterStemmer())


def test_get_punctuation_1(parser_default):
    punctuations = parser_default.punctuations
    testing_punctuation = ","
    assert testing_punctuation in punctuations


def test_get_punctuation_2(parser_default):
    punctuations = parser_default.punctuations
    testing_punctuation = "/"
    assert testing_punctuation in punctuations


def test_get_punctuation_3(parser_default):
    punctuations = parser_default.punctuations
    testing_punctuation = "("
    assert testing_punctuation in punctuations


def test_get_punctuation_4(parser_default):
    punctuations = parser_default.punctuations
    testing_punctuation = ")"
    assert testing_punctuation in punctuations


def test_get_punctuation_5(parser_default):
    punctuations = parser_default.punctuations
    testing_punctuation = "（"
    assert testing_punctuation in punctuations


def test_stopwords_file_path_english():
    language = "english"
    parser = Parser(language=language)
    true_path = os.path.join(os.getcwd(), "ir", "stopwords", "EnglishStopwords.txt")
    assert parser._get_stopwords_file_path() == true_path


def test_remove_stopwords_1(parser_default):
    words_list = ["running", "I", "is", "fun"]
    assert parser_default.remove_stopwords(words_list) == ["running", "fun"]


def test_remove_stopwords_2(parser_default):
    words_list = ["them", "he", "she", "fun"]
    assert parser_default.remove_stopwords(words_list) == ["fun"]


def test_tokenise_english(parser_porter):
    words_list = "she is swimming for fun while he is running"
    true_list = ["she", "is", "swim", "for", "fun", "while", "he", "is", "run"]
    assert parser_porter.tokenise(words_list) == true_list


def test_stem_1(parser_porter):
    word = "running"
    assert parser_porter.stem(word) == "run"


def test_stem_2(parser_porter):
    word = "swimming"
    assert parser_porter.stem(word) == "swim"

def test_clean_punctuation(parser_default):
    string = "Hello, world!"
    assert parser_default._clean_punctuation(string) == "Hello world"


