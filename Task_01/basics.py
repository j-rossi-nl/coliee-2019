import pandas as pd
import pytrec_eval
import numpy as np
import logging
import re
import pickle
import scipy.sparse as sp
import importlib
import tqdm

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize

from langdetect import detect as lang_detect
from langdetect.lang_detect_exception import LangDetectException

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.sparse import csr_matrix, lil_matrix

from gensim.summarization.summarizer import summarize

from imblearn.over_sampling import SMOTE
from collections import Counter
from cacheout.lru import LRUCache
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool


def language_detect(s):
    try:
        return lang_detect(s)
    except LangDetectException:
        return 'non-en'


def parallelize(data, func, num_of_processes):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func)


def parallelize_on_rows(data, func, num_of_processes):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


class ColieeData:
    nb_cases = 0
    nb_candidates_per_case = 0

    def __init__(self, filename):
        self.data = pd.read_csv(filename, usecols=['case_id', 'case_text', 'candidate_id', 'candidate_text',
                                                   'candidate_is_noticed'])
        self.data.sort_values(by=['case_id', 'candidate_id'], ascending=[True, True], inplace=True)
        self.update_class_var()

        # All of those will be populated by prepare_data
        self.stacked_cases = None
        self.unique_cases = None

        # The candidate cases
        self.candidate_cases = None

        # The corpus is stacked :
        # NB_CASES * a list of the NB_CANDIDATES_PER_CASE texts for the candidate cases of this query case
        # followed by NB_CASES texts for the query cases
        self.corpus = None

        # The qrels judgements
        # a 2-D array of shape (NB_CASES, NB_CANDIDATES_PER_CASE)
        self.qrels = None

        # Prepare
        self.__prepare_data__()

    def update_class_var(self):
        ColieeData.nb_cases = len(self.data['case_id'].unique())

        assert len(self.data) % ColieeData.nb_cases == 0
        ColieeData.nb_candidates_per_case = int(len(self.data) / self.nb_cases)


    def __prepare_data__(self):
        """
        Starting from the available data, create subsets
        :return:
        """
        # The cases
        self.stacked_cases = self.data['case_text']
        self.unique_cases = pd.Series(self.stacked_cases.unique())

        # The candidate cases
        self.candidate_cases = self.data['candidate_text']

        # The corpus is stacked :
        # NB_CASES * a list of the NB_CANDIDATES_PER_CASE texts for the candidate cases of this query case
        # followed by NB_CASES texts for the query cases
        self.corpus = self.candidate_cases.append(self.unique_cases)

        # The qrels judgements
        # a 2-D array of shape (NB_CASES, NB_CANDIDATES_PER_CASE)
        self.qrels = self.data['candidate_is_noticed'].values.reshape(ColieeData.nb_cases,
                                                                      ColieeData.nb_candidates_per_case)

    def apply_preprocessing(self, preprocess):
        """
        Apply preprocess to the texts.

        :param preprocess: a callable that accepts a text and returns a preprocessed text
        :return:
        """
        POOL_SIZE = 32

        self.data['case_text'] = parallelize_on_rows(self.data['case_text'], preprocess, POOL_SIZE)
        self.data['candidate_text'] = parallelize_on_rows(self.data['candidate_text'], preprocess, POOL_SIZE)
        self.__prepare_data__()

    @staticmethod
    def split_input(X):
        """
        Split the input between candidate cases / query cases

        :param X: one the corpus representations
        :return: query cases as a (NB_CASES, NB_FEATURES) 2-D array
                 candidate cases as a list of NB_CASES of 2-D arrays (NB_CANDIDATES_PER_CASE, NB_FEATURES) 3-D array
        """
        cd = X[:ColieeData.nb_cases * ColieeData.nb_candidates_per_case]
        split_candidates = []
        for i in range(ColieeData.nb_cases):
            split_candidates.append(
                cd[i * ColieeData.nb_candidates_per_case:(i + 1) * ColieeData.nb_candidates_per_case])

        return X[-ColieeData.nb_cases:].reshape(ColieeData.nb_cases, -1), \
               split_candidates

    @staticmethod
    def stratify_input(X):
        """
        Prepare the data for ML. It is stratified in the way that it is a single matrix (NB_CASES*NB_CANDIDATES_PER_CASE, 2*NB_FEATURES)
        where the rows are arranged in the following way: there are NB_CASES blocks of NB_CANDIDATES_PER_BLOCK rows, each row
        having the NB_FEATURES of the case vector concatenated with the NB_FEATURES of the candidate vector.

        :param X: one of the corpus representations
        :return: a single stratified matrix with shape (NB_CASES*NB_CANDIDATES_PER_CASE, 2*NB_FEATURES)
        """
        cs, cd = ColieeData.split_input(X)

        # Different operations if X is sparse or dense matrix
        if isinstance(X, csr_matrix):
            # Sparse features
            lil_cs = cs.tolil()
            lil_stratified_cases = lil_matrix((ColieeData.nb_cases*ColieeData.nb_candidates_per_case, cs.shape[1]))
            for i in range(ColieeData.nb_cases):
                for j in range(ColieeData.nb_candidates_per_case):
                    lil_stratified_cases.rows[i*ColieeData.nb_candidates_per_case + j] = lil_cs.rows[i].copy()
                    lil_stratified_cases.data[i * ColieeData.nb_candidates_per_case + j] = lil_cs.data[i].copy()

            csr_stratified_candidates = sp.vstack(cd)
            return sp.hstack([lil_stratified_cases, csr_stratified_candidates], format='csr')
        else:
            # Dense features
            stratified_cases = np.zeros(((ColieeData.nb_cases*ColieeData.nb_candidates_per_case, cs.shape[1])))
            for i in range(ColieeData.nb_cases):
                for j in range(ColieeData.nb_candidates_per_case):
                    stratified_cases[i * ColieeData.nb_candidates_per_case + j] = cs[i].copy()
            stratified_candidates = np.vstack(cd)
            return np.hstack([stratified_cases, stratified_candidates])


class ColieePreprocessor:
    """
    Some preprocessing functions
    """
    MIN_SENTENCE_LENGTH = 5
    MAX_TEXT_LENGTH = 180

    @staticmethod
    def do_nothing(text):
        """
        For test purposes...
        :param text:
        :return:
        """
        return text

    @staticmethod
    def clean(text):
        """
        Cleaning : remove extra '\n', extra spaces, remove non-english sentences
        :param text:
        :return:
        """
        # Texts are full of \n
        clean = text.replace('\n', ' ')
        clean = re.sub(r'[ ]{2,}', ' ', clean)

        sentences = [s for s in sent_tokenize(clean) if len(s.split()) > ColieePreprocessor.MIN_SENTENCE_LENGTH]
        only_en = ' '.join([s for s in sentences if language_detect(s) == 'en'])
        return only_en

    @staticmethod
    def summarize(text):
        """
        Use a summarization technique to reduce the size of the text.

        :param text:
        :return:
        """
        clean = ColieePreprocessor.clean(text)
        summary = summarize(text=clean, word_count=ColieePreprocessor.MAX_TEXT_LENGTH)
        summary = summary if len(summary)>0 else clean
        return summary


class ColieeVectorizer:
    """
    This class is helper for all text representation of the COLIEE Task 01 corpus.
    """
    default_parameters = {
        'ngram_range': (1, 1),
        'min_df': 5,
        'max_df': 0.8,
        'max_features': None,
        'lsa_components': 20,
        'lsa_based_on': 'tfidf',
        'bert_server': 'localhost'
    }

    representations = ('raw', 'bow', 'tfidf', 'lsa', 'bert')

    def __init__(self, data: ColieeData, parameters=None):
        """
        Instantiate a Vectorizer

        :param data: a ColieeData object
        :param args: Arguments for the CountVectorizer, TFIDFVectorizer, TruncatedSVD. It is a dictionary.
                     Relevant keys are 'lsa_components', 'ngram_range', 'min_df', 'max_df', 'max_features'
        """
        self.data = data
        self.parameters = parameters if parameters is not None else ColieeVectorizer.default_parameters

        # Which methods to use for which representation
        # Should have the same keys as self.vector_representations
        self.build_representation = {
            'raw':   self.prepare_raw,
            'bow':   self.prepare_bow,
            'tfidf': self.prepare_tfidf,
            'lsa':   self.prepare_lsa,
            'bert':  self.prepare_bert
        }
        self.vector_representations = {}

        self.logger = logging.getLogger('ColieeVectorizer')

    def prepare_all(self, methods=None, savetofile=None):
        """

        :return: a dictionary of vectorization methods and the corresponding vectors
        """
        use_methods = methods if methods is not None else ColieeVectorizer.representations
        for m in use_methods:
            self.build_representation[m]()
            if savetofile:
                self.to_file(savetofile)

    def prepare_raw(self):
        self.vector_representations['raw'] = self.data.corpus.values

    def prepare_bow(self, overwrite=False):
        """
        Run the CountVectorizer with parameters set in the constructor.
        The term-doc matrix is stored in self.vector_corpus_representations['bow'], and
        self.vector_split_representations

        :param overwrite: if a BoW was already prepared, should we overwrite?
        :return: nothing
        """
        self.logger.info('Preparing the BoW')
        if 'bow' in self.vector_representations and overwrite is False:
            self.logger.info('BoW already there, no overwrite')
            return

        # Prepare stemming with cache
        stemmer = PorterStemmer()
        cache = LRUCache(maxsize=2 ** 16)

        @cache.memoize()
        def stem_word(word):
            return stemmer.stem(word)

        # Our analyzer will proceed with tokenizing by regexp (all groups of more than 2 alpha characters)
        # and then will stem the words
        analyzer = CountVectorizer(analyzer='word', token_pattern='[A-Za-z]{2,}').build_analyzer()

        def stemming_analyzer(doc):
            return (stem_word(w) for w in analyzer(doc))

        bow = CountVectorizer(
            stop_words='english',
            lowercase=True,
            analyzer=stemming_analyzer,
            min_df=self.parameters['min_df'],
            max_df=self.parameters['max_df'],
            ngram_range=self.parameters['ngram_range'],
            max_features=self.parameters['max_features']
        )

        vecs = bow.fit_transform(raw_documents=self.data.corpus)
        self.logger.info('BoW with {} documents, {} terms'.format(vecs.shape[0],
                                                                  vecs.shape[1]))
        self.vector_representations['bow'] = vecs

    def prepare_tfidf(self, overwrite=False):
        """
        Run the TFIDFTransformer over the BoW of the corpus. If the BoW was not created, by prepare_bow, it will be
        created first. The parameters of the vectorizer are those given to the constructor.
        The TFIDF term-doc matrix is stored in self.corpus_tfidf

        :param overwrite: if tfidf was already prepared, should we overwrite?
        :return: nothing
        """
        # If already present, should we overwrite ?
        self.logger.info('Preparing the TF-IDF')
        if 'tfidf' in self.vector_representations and overwrite is False:
            self.logger.info('TF-IDF already there, no overwrite')
            return

        # If not already done, get the BoW
        self.build_representation['bow']()
        tfidf = TfidfTransformer()
        vecs = tfidf.fit_transform(X=self.vector_representations['bow'])
        self.logger.info('TF-IDF with {} documents, {} terms'.format(vecs.shape[0],
                                                                     vecs.shape[1]))
        self.vector_representations['tfidf'] = vecs

    def prepare_lsa(self, overwrite=False):
        """
        Run a Dimensionality Reduction SVD.
        If needed, the Bow / TFIDF term-doc matrix will be computed before. The parameters are given to the constructor.

        :param overwrite: if lsa was already prepared, should we overwrite?
        :return: nothing
        """
        assert self.parameters['lsa_based_on'] in ['bow', 'tfidf']

        # If already present, should we overwrite ?
        self.logger.info('Preparing the LSA')
        if 'lsa' in self.vector_representations and overwrite is False:
            self.logger.info('LSA already there, no overwrite')
            return

        self.build_representation[self.parameters['lsa_based_on']]()
        use_corpus = self.vector_representations[self.parameters['lsa_based_on']]

        svd = TruncatedSVD(n_components=self.parameters['lsa_components'])
        vecs = svd.fit_transform(X=use_corpus)
        self.logger.info('LSA with {} documents, {} features'.format(vecs.shape[0],
                                                                     vecs.shape[1]))
        self.vector_representations['lsa'] = vecs

    def prepare_bert(self, overwrite=False):
        """
        Use BERT to create semaintic vectors for documents.

        :param overwrite: if bert was already prepared, should we overwrite?
        :return: nothing
        """
        # No direct import - Had to do it, to help with pickling
        # If 'from bert_serving.client import BertClient' then it tries to pickle a Thread Lock
        bert_serving_client = importlib.import_module('bert_serving.client')
        BertClient = getattr(bert_serving_client, 'BertClient')

        # Use BERT client to get embeddings for the sentence
        # The server has a configuration for POOLING, returns 1 embedding per text
        # Can raise a TimeoutError
        bc = BertClient(ip=self.parameters['bert_server'], timeout=60000)


        # If already present, should we overwrite ?
        self.logger.info('Preparing BERT vectors')
        if 'bert' in self.vector_representations and overwrite is False:
            self.logger.info('BERT already there, no overwrite')
            return
        corpus = self.data.corpus
        BATCH_SIZE = 64

        def batchify(data):
            for i in range(len(data) // BATCH_SIZE + 1):
                begin, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
                if end >= len(data):
                    end = len(data) - 1
                batch = data[begin:end]
                yield bc.encode(list(batch))

        vecs = np.vstack([x for x in batchify(corpus)])
        self.vector_representations['bert'] = vecs
        self.logger.info('BERT with {} documents, {} features'.format(vecs.shape[0],
                                                                      vecs.shape[1]))

    def to_file(self, filename):
        """
        Save this Vectorizer to file, including all the prepared representations

        :param filename: file
        :return: nothing
        """
        self.logger.info('Saved to file {}'.format(filename))
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def from_file(filename):
        v = pickle.load(open(filename, 'rb'))
        v.data.update_class_var()
        return v


class Ranker(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def from_file(filename):
        """
        Load an object from file

        :param filename:
        :return:
        """
        object = pickle.load(open(filename, 'rb'))
        object.logger.info('Loading from file {}'.format(filename))
        return object

    def to_file(self, filename):
        """
        Save the object to file

        :param filename:
        :return:
        """
        self.logger.info('Saving model to file {}'.format(filename))
        pickle.dump(self, open(filename, 'wb'))

    @abstractmethod
    def rank(self, X, Y):
        """
        The ranker. X is the vector representing the query (an array (1, NB_FEATURES), and Y is the
        matrix of all potential matches. It returns an array of scores for all vectors of Y
        It is an abstract method, it has to be implemented in a sub-class

        :param X: Query vector
        :param Y: Collection vectors
        :return: Vector of scores
        """
        pass


class DirectRanker(Ranker):
    """
    When ranking score is computed directly from vector features.
    """
    @abstractmethod
    def get_scoring_function(self):
        """
        Abstract method, to be implemented by subclasses
        :return: a callable like cosine_similarity that takes vectors as inputs and computes a score
        """

    def rank(self, X, Y):
        """
        Implements a ranker based on cosine similarity

        :param X:
        :param Y:
        :return:
        """
        scorer = self.get_scoring_function()
        return scorer(X=X, Y=Y).reshape(-1,)


class ClassifierRanker(Ranker):
    """
    Implements a relevance score based on a trained Classifier model
    """
    def __init__(self):
        super().__init__()
        self.logger.info('Building model from scratch')
        self.model = None

    @abstractmethod
    def create_model(self):
        """
        Implement this function in subclasses to create the sklearn model
        :return: an estimator that is a classifier (see sklearn is_classifier
        """
        pass

    @abstractmethod
    def create_param_grid(self):
        """
        Implement this function in subclasses to create the parameter grid
        :return: parameter grid
        """
        pass

    def train_model(self, X, Y):
        """
        Train a model.

        :param X: a 2-D array (NB_CASES * NB_CANDIDATES_PER_CASE, 2*NB_FEATURES)
        :param Y: a 1-D array (NB_CASES * NB_CANDIDATES_PER_CASE) with the labels
        :return: nothing
        """
        # We will draw the cases that go to the train set, and those who go to the test set
        # So not all cases are in the train set, and the test set is unseen data
        self.logger.info('Training a Classifier')
        case_ids = list(range(ColieeData.nb_cases))
        train_ids, test_ids = train_test_split(case_ids, test_size=0.25)

        self.logger.info('Preparing the data')
        X_cases = [X[i * ColieeData.nb_candidates_per_case:(i + 1) * ColieeData.nb_candidates_per_case] for i in
                   range(ColieeData.nb_cases)]
        Y_cases = [Y[i * ColieeData.nb_candidates_per_case:(i + 1) * ColieeData.nb_candidates_per_case] for i in
                   range(ColieeData.nb_cases)]

        X_train = []
        Y_train = []
        for train_id in train_ids:
            X_train.append(X_cases[train_id])
            Y_train.append(Y_cases[train_id])

        X_test = []
        Y_test = []
        for test_id in test_ids:
            X_test.append(X_cases[test_id])
            Y_test.append(Y_cases[test_id])

        Y_train_array = np.concatenate(Y_train)
        Y_test_array = np.concatenate(Y_test)

        if isinstance(X, csr_matrix):
            # Sparse Features
            X_train_array = sp.vstack(X_train)
            X_test_array = sp.vstack(X_test)
        else:
            # Dense Features
            X_train_array = np.vstack(X_train)
            X_test_array = np.vstack(X_test)

        # The dataset will be unbalanced, we need some re-sampling
        self.logger.info("Re-sampling data")
        ros = SMOTE(random_state=0)
        X_test_resampled, Y_test_resampled = ros.fit_resample(X_test_array, Y_test_array)
        X_train_resampled, Y_train_resampled = ros.fit_resample(X_train_array, Y_train_array)
        self.logger.info('Training Data: {}'.format(sorted(Counter(Y_train_resampled).items())))
        self.logger.info('Test     Data: {}'.format(sorted(Counter(Y_test_resampled).items())))

        estimator = self.create_model()
        param_grid = self.create_param_grid()

        self.logger.info('Grid Search with param_grid : {}'.format(param_grid))
        self.model = GridSearchCV(estimator, param_grid, cv=5)
        self.model.fit(X=X_train_resampled, y=Y_train_resampled)

        tn, fp, fn, tp = confusion_matrix(Y_test_resampled, self.model.predict(X_test_resampled)).ravel()

        self.logger.info('Score on Test set : {:.2f}'.format(self.model.score(X=X_test_resampled, y=Y_test_resampled)))
        self.logger.info('Confusion Matrix : [ {:6d} {:6d} ]'.format(tn, fp))
        self.logger.info('Confusion Matrix : [ {:6d} {:6d} ]'.format(fn, tp))

    def rank(self, X, Y):
        """

        :param X: a query vector of shape (1, NB_FEATURES)
        :param Y: a candidate matrix of shape (NB_CANDIDATES_PER_CASE, NB_FEATURES)
        :return: a score vector (1, NB_CANDIDATES_PER_CASE)
        """
        if isinstance(X, csr_matrix):
            X_input = sp.vstack([X]*ColieeData.nb_candidates_per_case)
            X_input = sp.hstack([X_input, Y])
        else:
            X_input = np.vstack([X]*ColieeData.nb_candidates_per_case)
            X_input = np.hstack([X_input, Y])

        # Just use the prediction of the classifier as a score
        return self.model.predict(X_input)


class RankingBy:
    """
    In this class, we use the vector representation to compute relevance score with the provided Ranker object.
    In this methdd, we're in the case of labelled data.
    """
    @staticmethod
    def evaluate_vectorizer(vectorizer : ColieeVectorizer, ranker : Ranker, methods=ColieeVectorizer.representations):
        """
        Use a ColieeVectorizer as input, and uses all vectors.

        :param vectorizer: a ColieeVectorizer
        :param methods: the list of methods of vectorization to use, or 'all'
        :param ranker: an object of a subclass of Ranker
        :return: a dictionary with results
        """
        logger = logging.getLogger('RankingBy')

        qrels = vectorizer.data.qrels
        runs = {}
        vectorizer.prepare_all(methods=methods)
        for method in methods:
            v = vectorizer.vector_representations[method]
            cs, cd = ColieeData.split_input(v)
            runs[method] = {'cases': cs, 'candidates': cd}

        results= {}
        for k,v in runs.items():
            logger.info('Evaluating for {} vectors'.format(k))
            results[k] = RankingBy.evaluate(
                cases_vec=v['cases'],
                candidates_vec=v['candidates'],
                qrels=qrels,
                ranker=ranker
            )

        return results

    @staticmethod
    def evaluate(ranker, cases_vec, candidates_vec, qrels):
        """
        Use the ranker function between the representations of cases and their candidate cases, as a measure of relevance
        The resulting ranking is then evaluated based on average Recall and Precision.
        See ColieeVectorizer.split_input method to get proper arrays as input for this method

        :param ranker: the ranker of class Ranker
        :param cases_vec: The representations for the cases, a matrix NB_CASES * NB_FEATURES
        :param candidates_vec: The representations for the candidate cases. It is a list of length NB_CASES,
                               each element is a matrix NB_CANDIDATE_CASES_PER_CASE * NB_FEATURES
        :param qrels: the relevance judgements, an array of shape NB_CASES * NB_CANDIDATES_PER_CASE
        :return: the results as a dictionary
        """
        assert cases_vec.shape[0] == ColieeData.nb_cases
        assert len(candidates_vec) == ColieeData.nb_cases
        assert all([c.shape[0] == ColieeData.nb_candidates_per_case for c in candidates_vec])

        scores_df = pd.DataFrame(columns=['case_id', 'candidate_id', 'score', 'qrel'])
        for i in range(ColieeData.nb_cases):
            case_id = i + 1
            case_vec = cases_vec[i]
            case_candidates_vec = candidates_vec[i]
            scores = ranker.rank(X=case_vec.reshape(1, -1), Y=case_candidates_vec)
            scores_df = scores_df.append(
                pd.DataFrame(data=list(zip(
                    [case_id] * ColieeData.nb_candidates_per_case,
                    range(1, ColieeData.nb_candidates_per_case + 1),
                    scores,
                    qrels[i])),
                    columns=['case_id', 'candidate_id', 'score', 'qrel']),
                ignore_index=True
            )

        return RankingEvaluation.use_pytrec(scores_df)

    @staticmethod
    def rank_vectorizer(vectorizer : ColieeVectorizer, ranker : Ranker, methods=ColieeVectorizer.representations):
        """
        Use a ColieeVectorizer as input, and uses all vectors.

        :param vectorizer: a ColieeVectorizer
        :param methods: the list of methods of vectorization to use, or 'all'
        :param ranker: an object of a subclass of Ranker
        :return: a dictionary with results
        """
        logger = logging.getLogger('RankingBy')

        runs = {}
        vectorizer.prepare_all(methods=methods)
        for method in methods:
            v = vectorizer.vector_representations[method]
            cs, cd = ColieeData.split_input(v)
            runs[method] = {'cases': cs, 'candidates': cd, 'case_ids': vectorizer.data.data['case_id'].unique()}

        results= {}
        for k,v in runs.items():
            logger.info('Evaluating for {} vectors'.format(k))
            results[k] = RankingBy.rank(
                ranker=ranker,
                cases_vec=v['cases'],
                candidates_vec=v['candidates'],
                case_ids=v['case_ids']
            )

        return results

    @staticmethod
    def rank(ranker, cases_vec, candidates_vec, case_ids):
        """
        Use the ranker function between the representations of cases and their candidate cases, as a measure of relevance
        The resulting ranking is then evaluated based on average Recall and Precision.
        See ColieeVectorizer.split_input method to get proper arrays as input for this method

        :param ranker: the ranker of class Ranker
        :param cases_vec: The representations for the cases, a matrix NB_CASES * NB_FEATURES
        :param candidates_vec: The representations for the candidate cases. It is a list of length NB_CASES,
                               each element is a matrix NB_CANDIDATE_CASES_PER_CASE * NB_FEATURES
        :param qrels: the relevance judgements, an array of shape NB_CASES * NB_CANDIDATES_PER_CASE
        :return: the results as a dictionary
        """
        scores_df = pd.DataFrame(columns=['case_id', 'candidate_id', 'score', 'qrel'])
        for i, case_id in tqdm.tqdm(enumerate(case_ids), total=len(case_ids)):
            case_vec = cases_vec[i]
            case_candidates_vec = candidates_vec[i]
            scores = ranker.rank(X=case_vec.reshape(1, -1), Y=case_candidates_vec)
            scores_df = scores_df.append(
                pd.DataFrame(data=list(zip(
                    [case_id] * len(case_candidates_vec),
                    range(1, len(case_candidates_vec) + 1),
                    scores)),
                    columns=['case_id', 'candidate_id', 'score']),
                ignore_index=True
            )

        return scores_df





class RankingEvaluation:
    """
    A helper using pytrec_eval.

    """
    @staticmethod
    def use_pytrec(scores_df):
        """
        Computes the average value for Recall and Precision
        The average is micro (average of the Recall / Precision observed on each case).
        Removes all measures with a cut >= NB_CASES.
        Computes as well F1-Scores

        :param scores_df: the scores and qrels. A pandas DataFrame, with columns
                          'case_id', 'candidate_id', 'score', 'qrel'
        :return: A dictionary of results
        """
        # A helper function to allow the usage of pytrec_eval
        def recur_dictify(frame):
            """
            Take a pandas DataFrame and returns a nested dictionary.
            For example:
            A   B   C
            1   1   1
            1   2   2
            1   3   1
            2   1   1

            The nested dictionary is then:
            {
                1: {1:1, 2:2, 3:1},
                2: {1:1}
            }

            :param frame: a pandas dataframe
            :return: nested dictionary
            """
            if len(frame.columns) == 1:
                if frame.values.size == 1: return frame.values[0][0]
                return frame.values.squeeze()
            grouped = frame.groupby(frame.columns[0])
            d = {k: recur_dictify(g.iloc[:, 1:]) for k, g in grouped}
            return d

        def dict_with_str_keys_int_values(d):
            """
            Given a dictionary, converts all keys to string, and all values to int
            :param d: dictionary
            :return: dictionary
            """
            return {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in d.items()}

        def dict_with_str_keys_float_values(d):
            """
            Given a dictionary, converts all keys to string, and all values to float
            :param d: dictionary
            :return: dictionary
            """
            return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in d.items()}

        qrel_dict = recur_dictify(scores_df[scores_df['qrel'] == 1][['case_id', 'candidate_id', 'qrel']])
        qrel_dict = dict_with_str_keys_int_values(qrel_dict)

        run_dict = recur_dictify(scores_df[['case_id', 'candidate_id', 'score']])
        run_dict = dict_with_str_keys_float_values(run_dict)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, {'recall', 'P'})
        results_dict = evaluator.evaluate(run_dict)

        # remove all measures with cut >= NB_CANDIDATES_PER_CASE
        cut_re = re.compile(r'[A-Za-z]+_(?P<cut>[0-9]+)')
        remove = [k for k,_ in results_dict['1'].items() if int(cut_re.findall(k)[0]) >= ColieeData.nb_candidates_per_case]
        for key in remove:
            for _,v in results_dict.items():
                del v[key]

        # compute F1-score
        cuts = [cut_re.findall(k)[0] for k,_ in results_dict['1'].items()]
        for _,v in results_dict.items():
            for cut in cuts:
                P = v['P_{}'.format(cut)]
                R = v['recall_{}'.format(cut)]
                v['f1_{}'.format(cut)] = 2*P*R/(P+R) if P+R>0 else 0.0

        averages = {measure: np.mean([v[measure] for _, v in results_dict.items()]) for measure in
                    [k for k, _ in results_dict['1'].items()]}
        return averages

    @staticmethod
    def print_results(results):
        """
        Nice display of results

        :param results:  A nested dictionary {'method 1': {'measure 1': xx, 'measure 2': yy},
                                              'method 2': {'measure 1': zz, 'measure 2': aa}
        :return: nothing
        """
        print('\n')
        for method, results in results.items():
            print('RESULTS for : {}'.format(method.upper()))
            for k, v in results.items():
                print('{:10}: {:.2f}'.format(k.upper(), v))
            print('\n\n')
