import tensorflow as tf
import tensorflow_hub as hub
import os
import bert

from bert import run_classifier
from bert import tokenization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'


class BertEncoder:
    """
    This will get a pre-trained BERT
    """

    # This is a path to an uncased (all lowercase) version of BERT
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    SPLIT_ID = 'case_id'
    DATA_COLUMN_A = 'case_text'
    DATA_COLUMN_B = 'candidate_text'
    LABEL_COLUMN = 'candidate_is_noticed'
    LABEL_LIST = (0, 1)
    # We'll set sequences to be at most 512 tokens long.
    MAX_SEQ_LENGTH = 512

    def __init__(self):
        super().__init__()

    @staticmethod
    def __create_model__(input_ids, input_mask, segment_ids):
        """Creates a classification model."""
        bert_module = hub.Module(
            BertEncoder.BERT_MODEL_HUB,
            trainable=False)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        return bert_outputs["pooled_output"]

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    @staticmethod
    def __model_fn_builder__(num_labels):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]

            embeddings = BertEncoder.__create_model__(input_ids, input_mask, segment_ids)

            return tf.estimator.EstimatorSpec(mode, predictions=embeddings)

        # Return the actual model function in the closure
        return model_fn

    @staticmethod
    def __create_tokenizer_from_hub_module__():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(BertEncoder.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)


    @staticmethod
    def preprocessdata(X):
        """
        Transform the dataset into Bert-edible features.

        :param X: a Pandas DataFrame with columns 'case_id', 'case_text','candidate_text', 'candidate_is_noticed'
        :return:  a list of bert.run_classifier.InputFeatures
        """
        tokenizer = BertEncoder.__create_tokenizer_from_hub_module__()
        data = X
        data_InputExamples = data.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                   text_a=x[BertEncoder.DATA_COLUMN_A],
                                                                                   text_b=x[BertEncoder.DATA_COLUMN_B],
                                                                                   label=x[BertEncoder.LABEL_COLUMN] if BertEncoder.LABEL_COLUMN in x else 0), axis=1)
        data_features = bert.run_classifier.convert_examples_to_features(
            data_InputExamples,
            BertEncoder.LABEL_LIST,
            BertEncoder.MAX_SEQ_LENGTH, tokenizer
        )
        return data_features

    @staticmethod
    def encode(X):
        """
        Rank all candidate cases with regards to relevance to query case.
        X should contain NB_CANDIDATES_PER_CASE rows, each one having case_text the text of the query case, and
        candidate_text the text of the candidate case

        :param X: a dataframe with 2 columns : 'case_text', 'candidate_text'
        :return:
        """
        # Convert our train and test features to InputFeatures that BERT understands.
        data_features = BertEncoder.preprocessdata(X)

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig()
        model_fn = BertEncoder.__model_fn_builder__(num_labels=2)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": 64})

        # Create an input function
        predict_input_fn = bert.run_classifier.input_fn_builder(
            features=data_features,
            seq_length=BertEncoder.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        embeddings = list(estimator.predict(input_fn=predict_input_fn))
        return embeddings
