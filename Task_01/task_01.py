"""Task_01
Usage:
task_01 --text=CSV --save=SAVE --bertserver=ADDR
task_01 --vecs=SAVE --cosine
task_01 --vecs=SAVE --bm25 [--testrun=FILE]
task_01 --vecs=SAVE --logistic --logon=METHOD [--savelog=SAVE --loadlog=SAVE]
task_01 --vecs=SAVE --svm --svmon=METHOD [--savesvm=SAVE --loadsvm=SAVE]
task_01 --vecs=SAVE --bert --bertdir=DIR

Options:
  -h                  Show this screen
  --text=CSV          Read texts from a CSV file, Prepare the vector representations, Save the ColieeVectorizer on disk
  --vecs=SAVE         Read a ColieeVectorizer from a SAVE file, Use the representations to run some Rankers
  --save=FILE         Use FILE to save the ColieeVectorizer
  --bertserver=ADDR   Address of the Bert Server [default: localhost]
  --cosine            Run the cosine model
  --logistic          Run the Logistic Regression for Classification
  --logon=METHOD      Which text representation to use [default: lsa]
  --savelog=SAVE      Save the Logistic Regression model to disk
  --loadlog=SAVE      Load the Logistic Regression model from disk
  --svm               Run a SVM classification
  --svmon=METHOD      Which text representation to use [default: lsa]
  --savesvm=SAVE      Save the Logistic Regression model to disk
  --loadsvm=SAVE      Load the Logistic Regression model from disk
"""
import logging
import sys

from basics import ColieeData, ColieeVectorizer, RankingBy, RankingEvaluation
from baseline import CosineSimilarityRanker, LogisticRegressionRanker, SVMRanker, BM25Ranker

from docopt import docopt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)-8s %(name)-30s %(levelname)-8s %(message)s',
                        datefmt='%H:%M')
    logger = logging.getLogger('COLIEE_Task_01')
    args = docopt(__doc__, version='COLIEE v1.0')

    if args['--text'] is not None:
        # Read data
        input_file = args['--text']
        logger.info('Reading Data from file {}'.format(input_file))
        data = ColieeData(input_file)

        # Create all vectors
        logger.info('Creating Vectorizer')
        vectorizer = ColieeVectorizer(data=data)
        vectorizer.parameters['bert_server'] = args['--bertserver']
        vectorizer.prepare_all(savetofile=args['--save'])

    if args['--vecs'] is not None:
        # Read the ColieeVectorizer from file
        input_file = args['--vecs']
        logger.info('Reading ColieeVectorizer from file {}'.format(input_file))
        vectorizer = ColieeVectorizer.from_file(input_file)

        ranker = None
        methods = 'all'

        # Run the models
        if args['--cosine'] is True:
            logger.info('Baseline : Cosine Similarity')
            cosine = CosineSimilarityRanker()
            ranker = cosine
            methods = ['bow', 'tfidf', 'lsi', 'bert']

        if args['--bm25'] is True:
            logger.info('Baseline : BM25')
            bm25 = BM25Ranker()
            ranker = bm25
            methods = ['raw']
            if args['--testrun'] is not None:
                scores = RankingBy.rank_vectorizer(vectorizer=vectorizer, ranker=bm25, methods=methods)
                data = vectorizer.data.data[['case_id', 'candidate_id']]
                scores['raw'][['case_id', 'candidate_id', 'score']].to_csv(args['--testrun'], index=False)
                sys.exit(0)

        if args['--logistic'] is True:
            logger.info('Baseline : Logistic Regression')
            if args['--loadlog'] is not None:
                # Load from a saved model
                logistic = LogisticRegressionRanker.from_file(args['--loadlog'])
            else:
                # Build a new model and save it
                logistic = LogisticRegressionRanker()
                logistic.train_model(X=ColieeData.stratify_input(vectorizer.vector_representations[args['--logon']]),
                                     Y=vectorizer.data.qrels.flatten())
                if args['--savelog'] is not None:
                    logistic.to_file(args['--savelog'])
            ranker = logistic
            methods = args['--logon']

        if args['--svm'] is True:
            logger.info('Baseline : SVM')
            if args['--loadsvm'] is not None:
                # Load from a saved model
                svm = SVMRanker.from_file(args['--loadsvm'])
            else:
                # Build a new model and save it
                svm = SVMRanker()
                svm.train_model(
                    X=ColieeData.stratify_input(vectorizer.vector_representations[args['--svmon']]),
                    Y=vectorizer.data.qrels.flatten())
                if args['--savesvm'] is not None:
                    svm.to_file(args['--savesvm'])
            ranker = svm
            methods = args['--svmon']

        baseline = RankingBy.evaluate_vectorizer(vectorizer=vectorizer, ranker=ranker, methods=methods)
        RankingEvaluation.print_results(baseline)
