import numpy as np

from basics import DirectRanker, ClassifierRanker, Ranker

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from gensim.utils import tokenize
from gensim.corpora import Dictionary
from gensim.summarization.bm25 import BM25

class CosineSimilarityRanker(DirectRanker):
    """
    Implements a relevance score based on cosine similarity
    """
    def get_scoring_function(self):
        return cosine_similarity


class LogisticRegressionRanker(ClassifierRanker):
    def create_model(self):
        return LogisticRegression(penalty='l1', solver='saga', max_iter=1000)

    def create_param_grid(self):
        return {'C': [0.01]} #np.logspace(start=-6, stop=3, num=10)}


class SVMRanker(ClassifierRanker):
    def create_model(self):
        return LinearSVC(penalty='l2', loss='hinge', max_iter=10000)

    def create_param_grid(self):
        return {'C': np.logspace(start=-6, stop=-1, num=6)}


class BM25Ranker(Ranker):
    """
    Implements a relevance score based on BM25
    """
    def rank(self, X, Y):
        """
        Due to the way queries are done, we should instantiate the BM25 problem each time there is a query and a
        corpus.

        :param X: Query vector
        :param Y: Collection vectors
        :return: Vector of scores
        """
        def my_tokenizer(text):
            initial_tokens = list(tokenize(text, lowercase=True, deacc=True))
            filtered_tokens = [t for t in initial_tokens if len(t)>2]
            return filtered_tokens

        tokenized_corpus = [my_tokenizer(y) for y in Y]
        tokenized_query = my_tokenizer(X[0][0])
#        dictionary = Dictionary(documents=tokenized_corpus)
#        corpus = [dictionary.doc2bow(y) for y in tokenized_corpus]
#        query = dictionary.doc2bow(tokenized_query)

        bm = BM25(tokenized_corpus)
        average_idf = sum(map(lambda k: float(bm.idf[k]), bm.idf.keys())) / len(bm.idf.keys())
        scores = bm.get_scores(tokenized_query, average_idf)
        return scores
