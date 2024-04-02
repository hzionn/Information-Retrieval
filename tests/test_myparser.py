import os

from ir.myparser import Parser


def test_get_punctuation_1():
    punctuations = Parser().punctuations
    testing_punctuation = ","
    assert testing_punctuation in punctuations


def test_get_punctuation_2():
    punctuations = Parser().punctuations
    testing_punctuation = "/"
    assert testing_punctuation in punctuations


def test_get_punctuation_3():
    punctuations = Parser().punctuations
    testing_punctuation = "("
    assert testing_punctuation in punctuations


def test_get_punctuation_4():
    punctuations = Parser().punctuations
    testing_punctuation = ")"
    assert testing_punctuation in punctuations


def test_get_punctuation_5():
    punctuations = Parser().punctuations
    testing_punctuation = "（"
    assert testing_punctuation in punctuations


def test_stopwords_file_path_english():
    language = "english"
    parser = Parser(language=language)
    true_path = os.path.join(os.getcwd(), "ir", "stopwords", "EnglishStopwords.txt")
    assert parser._get_stopwords_file_path() == true_path


def test_stopwords_file_path_chinese():
    language = "chinese"
    parser = Parser(language=language)
    true_path = os.path.join(os.getcwd(), "ir", "stopwords", "ChineseStopwords.txt")
    assert parser._get_stopwords_file_path() == true_path


def test_remove_stopwords_1():
    parser = Parser()
    words_list = ["running", "I", "is", "fun"]
    assert parser.remove_stopwords(words_list) == ["running", "fun"]


def test_remove_stopwords_2():
    parser = Parser()
    words_list = ["them", "he", "she", "fun"]
    assert parser.remove_stopwords(words_list) == ["fun"]


def test_tokenise():
    parser = Parser()
    words_list = "she is swimming for fun while he is running"
    true_list = ["she", "is", "swim", "for", "fun", "while", "he", "is", "run"]
    assert parser.tokenise(words_list) == true_list


def test_stem_1():
    parser = Parser()
    word = "running"
    assert parser.stem(word) == "run"


def test_stem_2():
    parser = Parser()
    word = "swimming"
    assert parser.stem(word) == "swim"
