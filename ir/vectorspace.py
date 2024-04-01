import os
from pprint import pprint
from typing import List, Dict

from myparser import Parser
from model import TFIDF


class VectorSpace:
    def __init__(self, model=TFIDF(), parser=Parser()):
        self.model = model
        self.parser = parser
        self.documents_directory = ""
        self.documents_name_content = {}
        print("vector space initailized")

    def build(self, documents_directory: str):
        """build up our vector space model"""
        self.documents_directory = documents_directory
        self.documents_name_content = self._get_documents_name_content()
        self.documents_name_content = self._clean_all_documents()
        self.documents_name_content = self._sort_documents_by_size()
        pass

    def _clean_single_document(self, document_content: str) -> str:
        """clean single document content"""
        cleaned_content = self.parser.tokenise(document_content)
        cleaned_content = self.parser.remove_stopwords(cleaned_content)
        return " ".join(cleaned_content)
    
    def _clean_all_documents(self) -> Dict[str, str]:
        for name, content in self.documents_name_content.items():
            self.documents_name_content[name] = self._clean_single_document(content)
        return self.documents_name_content

    def _get_documents_name_content(self) -> Dict[str, str]:
        """get all documents' name and content mapping"""
        name_content = {}
        path = os.path.join(os.path.dirname(__file__), self.documents_directory)
        names = os.listdir(path)
        contents = []
        for document in names:
            with open(os.path.join(path, document), "r") as f:
                content = " ".join(f.readlines())
                contents.append(content)
        assert len(names) == len(contents)
        for name, content in zip(names, contents):
            name_content[name] = content
        return name_content
    
    def _sort_documents_by_size(self) -> Dict[str, str]:
        """sort documents_name_content by content size"""
        name_content = self.documents_name_content
        sorted_name_content = {}
        name_len = {name: len(name_content[name]) for name in name_content}
        name_len_sorted = dict(sorted(name_len.items(), key=lambda x: x[1], reverse=True))
        new_name_content = {name: name_content[name] for name in name_len_sorted.keys()}
        for doc, news in zip(list(new_name_content.keys()), list(new_name_content.values())):
            sorted_name_content[doc] = news
        return sorted_name_content
    

def main():
    vs = VectorSpace()
    vs.build("sample_data/EnglishNews")
    # print(vs.documents_name_content)
    print(vs.documents_name_content["News996.txt"])
    print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[0]]))
    print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[100]]))
    print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[200]]))
    print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[-1]]))
    # print(type(name_content["News996.txt"]))
    # parser = Parser()


if __name__ == "__main__":
    main()
