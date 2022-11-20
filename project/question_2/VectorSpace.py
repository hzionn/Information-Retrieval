from Parser import Parser
import util
import math
import numpy as np
from tqdm import tqdm
from nltk import pos_tag, word_tokenize


class VectorSpace:
    """ 
    An algebraic model for representing(tf-idf weighting) text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term.
    """
    def __init__(self, sample_news: list[str], documents: list[str] = []):
        self.documentVectors = []
        self.idf_values = {}
        self.parser = Parser()

        self.documents, self.sample_news = self.sort_out(documents=documents, sample_news=sample_news)

        print("...cleaning documents (tokenise, remove stopwords)...")
        self.cleaned_bloblist = []
        for doc in tqdm(self.documents):
            cleaned = self.parser.tokenise(doc)
            cleaned = self.parser.removeStopWords(cleaned)
            cleaned_doc = " ".join(cleaned)
            self.cleaned_bloblist.append(cleaned_doc)

        if len(self.cleaned_bloblist) > 0: self.build(self.cleaned_bloblist)

    def build(self, documents: list[str]):
        """Create the vector space for the passed document strings"""
        print("...get vector keyword index...", end="")
        self.vectorKeywordIndex: dict = self.getVectorKeywordIndex(documents)
        print(f"(found {len(self.vectorKeywordIndex)} vector keywords)...")
        print("...generating vectors for each document...")
        self.documentVectors = [self.makeVector(documents[i], i) for i in tqdm(range(len(documents)))]

    def getVectorKeywordIndex(self, documentList: list[str]) -> dict:
        """
        create the keyword associated to the position of the elements 
        within the document vectors
        """
        # Mapped documents into a single word string
        vocabularyString = " ".join(documentList)
        vocabularyList = vocabularyString.split()
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex = {}
        offset = 0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word] = offset
            offset += 1
        return vectorIndex  # {keyword: position}

    def makeVector(self, wordString: str, i: int) -> list[float]:
        """build document vector with tf-idf weighting"""
        vector: list[float] = [0] * len(self.vectorKeywordIndex)
        wordList = wordString.split()
        for word in wordList:
            if word in self.vectorKeywordIndex:
                vector[self.vectorKeywordIndex[word]] = self.tfidf(
                    word, wordList, self.documents, i)
        return vector

    def buildQueryVector(self, termList:list):
        """convert query string into a term vector"""
        query_string = " ".join(termList)
        cleaned = self.parser.tokenise(query_string)
        cleaned = self.parser.removeStopWords(cleaned)
        query_vector = self.makeVector(" ".join(cleaned), 0)
        return query_vector

    def related(self, doc_id: int = -1, faster_way: bool = True) -> list:
        """ 
        find documents that are related to the document indexed by passed index
        query documents will always be the LAST ONE
        """
        if not faster_way:
            print("slower calculation")
            ratings = [
                util.cosine(document_vector, self.documentVectors[doc_id])
                for document_vector in self.documentVectors]
        ratings = util.cosine(np.array(self.documentVectors), np.array(self.documentVectors[doc_id]))
        return ratings

    def search(self, searchList: list, distance: str, old_query_vector: list[float] = None) -> list:
        """search for documents that match based on a list of terms"""
        assert distance in ("cosine", "euclidean")
        
        print("...searching relevent documents...")
        queryVector = self.buildQueryVector(searchList)
        
        # for second retrieve
        if old_query_vector:
            # re-weighting with old query vector and new one to form a new query vector
            new_query_vector = (1 * np.array(old_query_vector)) + (0.5 * np.array(queryVector))
            ratings = util.cosine(np.array(self.documentVectors), new_query_vector)
            return ratings

        # for first retrieve
        if not old_query_vector:
            if distance == "cosine":
                ratings = util.cosine(np.array(self.documentVectors), np.array(queryVector))
            elif distance == "euclidean":
                ratings = util.euclidean(np.array(self.documentVectors), np.array(queryVector))
            return ratings, queryVector

    def sort_out(self, documents: list[str], sample_news: list[str]):
        """sort documents size from biggest to smallest for the sake of efficiency in calculating idf"""
        assert len(documents) == len(sample_news)
        print("...sort out documents size from biggest to smallest...")
        name_doc: dict[str, str] = {name: doc for doc, name in zip(documents, sample_news)}
        name_len = {name: len(name_doc[name]) for name in name_doc}
        name_len_sort = dict(sorted(name_len.items(), key=lambda x:x[1], reverse=True))
        new_name_doc = {name: name_doc[name] for name in name_len_sort.keys()}
        documents = [doc for doc in new_name_doc.values()]
        sample_news = [name for name in new_name_doc.keys()]
        return documents, sample_news

    def feedback(self, ratings_list: list, sample_news: list, old_query_vector: list[float]):
        """get pseudo feedback from top document and re-search"""
        top_ratings_index = np.argsort(ratings_list)[-10:][::-1]
        # top 1 document from first retrieved
        top_sample_new = sample_news[top_ratings_index[0]]

        # read top document
        with open(f"../EnglishNews/{top_sample_new}") as f:
            f = f.readlines()
            top_document = " ".join(f)
        cleaned = self.parser.tokenise(top_document)
        cleaned = self.parser.removeStopWords(cleaned)
        
        # keep only NN, VB in top document's then search again
        print("...get top document's pos tag...")
        new_query = [vocab for vocab, tag in pos_tag(cleaned) if tag in ("NN", "VB")]
        print("...re-search with feedback...")
        new_ratings = self.search(new_query, "cosine", old_query_vector=old_query_vector)
        return new_ratings


    def tf(self, word: str, blob):
        """term frequency"""
        count = 0
        for w in blob:
            if w == word:
                count += 1
        return count / len(blob)

    def n_containing(self, word: str, bloblist, i):
        """numbers of show up in all documents"""
        return sum(1 for blob in bloblist[i:] if word in blob)

    def idf(self, word: str, bloblist, i):
        """inverse document frequency"""
        if word not in self.idf_values:
            self.idf_values[word] = math.log(
                len(bloblist) / (1 + self.n_containing(word, bloblist, i)))
        return self.idf_values[word]

    def tfidf(self, word: str, blob, bloblist, i):
        return self.tf(word, blob) * self.idf(word, bloblist, i)


if __name__ == '__main__':

    # test data
    documents = [
        "The cat in the hat disabled",
        "A cat is a fine pet ponies.",
        "Dogs and cats make good pets.",
        "I haven't got a hat."
        ]

    vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)

    #print(vectorSpace.documentVectors)

    print(vectorSpace.related(1))

    #print(vectorSpace.search(["cat", "hat"]))