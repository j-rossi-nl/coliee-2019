"""bert_ranking
Usage:
bert_high_ranking --train ((--text=CSV | --inputs=SAVE) | (--traindata=CSV --testdata=CSV)) [--outputdir=DIR --batchsize=INT --epochs=INT]
bert_high_ranking --rank (--text=CSV | --inputs=SAVE) [--submission=FILE --cut=VALUE] [--rawfile=FILE] [--outputdir=DIR --batchsize=INT ] [--checkpoint=CKP]

Options:
  -h                  Show this screen
  --text=CSV          Read texts from a CSV file, will require preprocessing
  --saveinputs=SAVE   Save the InputExamples into a SAVE file (pickle)
  --test              Generate Inputs as one collection
  --inputs=SAVE       Read InputExamples from a SAVE file (pickle)
  --outputdir=DIR     Folder for BERT checkpoints [default: bert_output]
  --batchsize=INT     Size of training batches  [default: 8]
  --epochs=INT        Number of training epochs [default: 5]
  --rank              Use a trained BERT model for ranking
  --cut=VALUE         The score value at which to cut
  --submission=FILE   The file to submit to the competition
  --rawfile=FILE      File in which to save the raw scores
  --testcases=FILE    File in which to save which case_id were selected as test cases
  --checkpoint=CKP    Which checkpoint to use (CKP is a number)
"""
import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import bert
import pickle

from tensorflow.python.client import device_lib


from bert import run_classifier
from bert import optimization
from bert import tokenization

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from datetime import datetime
from docopt import docopt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'


class BertHighRanker():
    """
    This will get a pre-trained BERT, set it to not-trainable and add a DNN classifier with 1 hidden layer
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

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        tf.logging.set_verbosity(tf.logging.INFO)

    def create_model(self):
        pass

    def create_param_grid(self):
        pass

    @staticmethod
    def __create_model__(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
        """Creates a classification model."""
        bert_module = hub.Module(
            BertHighRanker.BERT_MODEL_HUB,
            trainable=True)
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
        output_layer = bert_outputs["pooled_output"]

        x = tf.layers.Dense(units=256, activation=tf.nn.relu, use_bias=True)(output_layer)
        x = tf.layers.Dropout(rate=0.1)(x)
        logits = tf.layers.Dense(units=2, activation=tf.nn.relu, use_bias=True)(x)
        probs = tf.nn.softmax(logits, axis=-1)

        with tf.variable_scope("loss"):
            predicted_labels = tf.argmax(probs, axis=-1, output_type=tf.int32) # shape [BATCH_SIZE, 1]

            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (tf.squeeze(predicted_labels), probs)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            loss = tf.losses.mean_pairwise_squared_error(labels=one_hot_labels,
                                                         predictions=probs)
            return (loss, predicted_labels, probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    @staticmethod
    def __model_fn_builder__(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, probs) = BertHighRanker.__create_model__(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(
                        label_ids,
                        predicted_labels)
                    auc = tf.metrics.auc(
                        label_ids,
                        predicted_labels)
                    recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                    precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                    true_pos = tf.metrics.true_positives(
                        label_ids,
                        predicted_labels)
                    true_neg = tf.metrics.true_negatives(
                        label_ids,
                        predicted_labels)
                    false_pos = tf.metrics.false_positives(
                        label_ids,
                        predicted_labels)
                    false_neg = tf.metrics.false_negatives(
                        label_ids,
                        predicted_labels)
                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, probs) = BertHighRanker.__create_model__(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                    'probabilities': probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    @staticmethod
    def __create_tokenizer_from_hub_module__():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(BertHighRanker.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)


    @staticmethod
    def preprocessdata(X, mode='train'):
        """
        Transform the dataset into Bert-edible features. If mode is 'train', then the data is split into
        train/test according to case_id, so that the cases in the test set are not represented in the train set.
        An over-samplind method is used to re-balance the datasets.
        When mode is not 'train', then the data is only transformed to features.

        :param X: a Pandas DataFrame with columns 'case_id', 'case_text','candidate_text', 'candidate_is_noticed'
        :param mode: What the data will be used for. If 'train', then it is split and oversampled. If not, then it is just prepared
        :return:  a 2-uple (train, test) of lists of bert.run_classifier.InputFeatures
        """
        tokenizer = BertHighRanker.__create_tokenizer_from_hub_module__()
        data = X

        if mode == 'train':
            # Split so that cases in the test set are not in the training set
            train_cases, test_cases = train_test_split(data['case_id'].unique(), train_size=0.75)
            train = data[data['case_id'].isin(train_cases)]
            test = data[data['case_id'].isin(test_cases)]

            ratio = 0.25    # Re-sample until we have the ratio for  Minority_Class / Majority Class
            ros = RandomOverSampler(random_state=0, ratio=ratio)
            train_resampled, _ = ros.fit_resample(train, train[BertHighRanker.LABEL_COLUMN])
            test_resampled, _ = ros.fit_resample(test, test[BertHighRanker.LABEL_COLUMN])

            train_df = pd.DataFrame(train_resampled, columns=data.columns.values)
            test_df = pd.DataFrame(test_resampled, columns=data.columns.values)

            train_InputExamples = train_df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                         text_a=x[BertHighRanker.DATA_COLUMN_A],
                                                                                         text_b=x[BertHighRanker.DATA_COLUMN_B],
                                                                                         label=x[BertHighRanker.LABEL_COLUMN]), axis=1)
            test_InputExamples = test_df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                       text_a=x[BertHighRanker.DATA_COLUMN_A],
                                                                                       text_b=x[BertHighRanker.DATA_COLUMN_B],
                                                                                       label=x[BertHighRanker.LABEL_COLUMN]), axis=1)

            train_features = bert.run_classifier.convert_examples_to_features(
                train_InputExamples,
                BertHighRanker.LABEL_LIST,
                BertHighRanker.MAX_SEQ_LENGTH, tokenizer
            )
            test_features = bert.run_classifier.convert_examples_to_features(
                test_InputExamples,
                BertHighRanker.LABEL_LIST,
                BertHighRanker.MAX_SEQ_LENGTH,
                tokenizer
            )

            return train_features, test_features, test_cases

        else:  # Data is prepared for inference
            data_InputExamples = data.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                       text_a=x[BertHighRanker.DATA_COLUMN_A],
                                                                                       text_b=x[BertHighRanker.DATA_COLUMN_B],
                                                                                       label=x[BertHighRanker.LABEL_COLUMN] if mode=='prepare' else 0), axis=1)
            data_features = bert.run_classifier.convert_examples_to_features(
                data_InputExamples,
                BertHighRanker.LABEL_LIST,
                BertHighRanker.MAX_SEQ_LENGTH, tokenizer
            )
            return data_features

    def train_model(self, X=None, train_data=None, test_data=None):
        """

        :param X: a dataframe with 3 columns : 'case_text', 'candidate_text', 'candidate_is_noticed', which is preprocessed,
                  or a 2-uple (train, test) of lists of bert.run_classifyer.InputFeatures
        :return:
        """
        # Convert our train and test features to InputFeatures that BERT understands.
        if X is not None:
            if isinstance(X, pd.DataFrame):
                train_features, test_features, _ = self.preprocessdata(X)
            else:
                train_features, test_features = X
        else:
            train_features = self.preprocessdata(train_data, 'prepare')
            test_features = self.preprocessdata(test_data, 'prepare')
        tf.logging.info('Finished with examples_to_features')

        # Compute train and warmup steps from batch size
        # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
        OUTPUT_DIR = self.parameters['--outputdir']
        BATCH_SIZE = int(self.parameters['--batchsize'])
        LEARNING_RATE = 1e-6
        NUM_TRAIN_EPOCHS = int(self.parameters['--epochs'])
        # Warmup is a period of time where hte learning rate
        # is small and gradually increases--usually helps training.
        WARMUP_PROPORTION = 0.1
        # Model configs
        SAVE_CHECKPOINTS_STEPS = 500
        SAVE_SUMMARY_STEPS = 100

        # Compute # train and warmup steps from batch size
        num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        # Run multi-GPU : train on all GPUs, evaluate on 1 GPU
        devices = device_lib.list_local_devices()
        gpu_names = [x.name for x in devices if x.device_type=='GPU']
        train_distribute = None
        eval_distribute = None
        if len(gpu_names) == 2:
            train_distribute = tf.contrib.distribute.OneDeviceStrategy(device=gpu_names[0])
            eval_distribute = tf.contrib.distribute.OneDeviceStrategy(device=gpu_names[1])

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
            model_dir=OUTPUT_DIR,
            save_summary_steps=SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
            keep_checkpoint_max=None,
            train_distribute=train_distribute,
            eval_distribute=eval_distribute)

        model_fn = BertHighRanker.__model_fn_builder__(
            num_labels=len(BertHighRanker.LABEL_LIST),
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": BATCH_SIZE})

        # Create an input function for training. drop_remainder = True for using TPUs.
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=BertHighRanker.MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=False)

        # Create an input function for training. drop_remainder = True for using TPUs.
        test_input_fn = bert.run_classifier.input_fn_builder(
            features=test_features,
            seq_length=BertHighRanker.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        #estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        #estimator.evaluate(input_fn=test_input_fn, steps=None)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, steps=None, throttle_secs=60, start_delay_secs=0)

        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    def rank(self, X, checkpoint=None):
        """
        Rank all candidate cases with regards to relevance to query case.
        X should contain NB_CANDIDATES_PER_CASE rows, each one having case_text the text of the query case, and
        candidate_text the text of the candidate case

        :param X: a dataframe with 2 columns : 'case_text', 'candidate_text'
        :return:
        """
        # Convert our train and test features to InputFeatures that BERT understands.
        if isinstance(X, pd.DataFrame):
            data_features = self.preprocessdata(X, mode='predict')
        else:
            data_features = X
        tf.logging.info('Finished with examples_to_features')

        # Compute train and warmup steps from batch size
        # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
        OUTPUT_DIR = self.parameters['--outputdir']
        BATCH_SIZE = int(self.parameters['--batchsize'])

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR)

        model_fn = BertHighRanker.__model_fn_builder__(
            num_labels=len(BertHighRanker.LABEL_LIST),
            learning_rate=1e-3, # FAKE, will not be used
            num_train_steps=1,  # FAKE, will not be used
            num_warmup_steps=1) # FAKE, will not be used

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": BATCH_SIZE})

        # Create an input function for training. drop_remainder = True for using TPUs.
        predict_input_fn = bert.run_classifier.input_fn_builder(
            features=data_features,
            seq_length=BertHighRanker.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        checkpoint_file = None
        if checkpoint is not None:
            checkpoint_file = os.path.join(OUTPUT_DIR, 'model.ckpt-{}'.format(checkpoint))

        tf.logging.info('Starting with Predicting')
        predictions = list(estimator.predict(input_fn=predict_input_fn, checkpoint_path=checkpoint_file))
        scores = [p['probabilities'][1] for p in predictions]
        return scores


def main():
    args = docopt(__doc__, version='COLIEE v1.0')

    if args['--train'] is True:
        br = BertHighRanker(parameters=args)
        if args['--inputs'] is not None or args['--text'] is not None:
            if args['--inputs'] is not None:
                train_data = pickle.load(open(args['--inputs'], 'rb'))
            else:
                train_data = pd.read_csv(args['--text'])
            br.train_model(X=train_data)
        else:
            train_data = pd.read_csv(args['--traindata'])
            test_data = pd.read_csv(args['--testdata'])
            br.train_model(train_data=train_data, test_data=test_data)

    if args['--rank'] is True:
        br = BertHighRanker(parameters=args)
        if args['--text'] is not None:
            test_data = pd.read_csv(args['--text'])
        else:
            test_data = pickle.load(open(args['--inputs'], 'rb'))

        checkpoint = None
        if args['--checkpoint'] is not None:
            checkpoint = int(args['--checkpoint'])
        scores = br.rank(test_data, checkpoint=checkpoint)
        test_data['score'] = scores
        if args['--rawfile'] is not None:
            test_data[['case_id', 'candidate_id', 'score']].to_csv(args['--rawfile'], index=False)

        if args['--submission'] is not None:
            test_data['run'] = 'ILPS_BERT'
            final_table = test_data.sort_values(by=['case_id', 'score'], ascending=[True, False])
            with open(args['--submission'], 'w') as submission:
                for _, v in final_table[final_table['score'] > float(args['--cut'])][['case_id', 'candidate_id', 'run']].iterrows():
                    submission.write('{:03d} {:03d} {}\n'.format(v['case_id'], v['candidate_id'], v['run']))


if __name__ == "__main__":
    main()
