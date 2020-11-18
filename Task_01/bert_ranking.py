"""bert_ranking
Usage:
bert_ranking --encode --text=CSV --saveinputs=SAVE [--balance]
bert_ranking --train (--hub | --local=DIR [--chkp=FILENAME]) ((--text=CSV | --inputs=SAVE) | (--traindata=CSV --evaldata=CSV)) [--outputdir=DIR --batchsize=INT --epochs=INT]
bert_ranking --rank (--text=CSV | --inputs=SAVE) [--submission=FILE --cut=VALUE] [--rawfile=FILE] [--outputdir=DIR --batchsize=INT ] [--checkpoint=CKP]

Options:
  -h                  Show this screen
  --encode            Create encoded features for train, eval or test
  --rank              Use a trained BERT model for ranking
  --train             Train a new BERT-based classifier for COLIEE
  --text=CSV          Read texts from a CSV file, will require preprocessing
  --saveinputs=SAVE   Save the InputExamples into a SAVE file (pickle)
  --inputs=SAVE       Read InputExamples from a SAVE file (pickle)
  --traindata=FILE    Training data, either a CSV or SAVE
  --evaldata=FILE     Evaluation data, either a CSV or SAVE
  --outputdir=DIR     Folder for BERT checkpoints [default: bert_output]
  --batchsize=INT     Size of training batches  [default: 8]
  --epochs=INT        Number of training epochs [default: 5]
  --cut=VALUE         The score value at which to cut
  --submission=FILE   The file to submit to the competition
  --rawfile=FILE      File in which to save the raw scores
  --checkpoint=CKP    Which checkpoint to use (CKP is a number)
  --hub               Restores a BERT pre-trained model from Tensorflow Hub
  --local=DIR         Restores a BERT pre-trained model from a local folder (copy vocab.txt and bert_config.json in the folder)
  --chkp=FILENAME     A specific checkpoint (xxx.chkp-yyyy part of the full name xxx.chkp-yyyy.data-...)
  --balance           Use over-sampling to rebalance the dataset (good for training data)
"""
import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import bert
import pickle
import sys

from tqdm import tqdm

from tensorflow.python.client import device_lib

from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


from docopt import docopt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'


class BertRanker():
    # This is a path to an uncased (all lowercase) version of BERT
    USE_HUB = True
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    USE_LOCAL = False
    LOCAL_DIR = ''
    CHKP_NAME = ''

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
    def __configure_class_vars__(parameters):
        if parameters['--local'] is not None:
            BertRanker.USE_HUB = False
            BertRanker.USE_LOCAL = True
            BertRanker.LOCAL_DIR = parameters['--local']
            if parameters['--chkp'] is not None:
              BertRanker.CHKP_NAME = parameters['--chkp']

    @staticmethod
    def __create_model__(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
        """Creates a classification model."""
        if BertRanker.USE_HUB:
            bert_module = hub.Module(
                BertRanker.BERT_MODEL_HUB,
                trainable=not is_predicting)
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
        else:
            bert_module = modeling.BertModel(
                config=modeling.BertConfig.from_json_file(os.path.join(BertRanker.LOCAL_DIR, 'bert_config.json')),
                is_training=not is_predicting,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=False)
            tvars = tf.trainable_variables()
            chkp_file = os.path.join(BertRanker.LOCAL_DIR, BertRanker.CHKP_NAME)
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, chkp_file)
            tf.train.init_from_checkpoint(chkp_file, assignment_map)
            output_layer = bert_module.get_pooled_output()

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probs = tf.nn.softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, probs)

            # If we're train/eval, compute loss between predicted and actual label
            #log_probs = tf.nn.log_softmax(logits, axis=-1)
            #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            # Try MSE to force irrelevant to be very low scores
            per_example_loss = tf.reduce_sum(tf.squared_difference(probs, one_hot_labels), axis=-1)
            loss = tf.reduce_mean(per_example_loss)
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

                (loss, predicted_labels, probs) = BertRanker.__create_model__(
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
                (predicted_labels, probs) = BertRanker.__create_model__(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                    'probabilities': probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    @staticmethod
    def __create_tokenizer__():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            if BertRanker.USE_HUB:
                bert_module = hub.Module(
                    BertRanker.BERT_MODEL_HUB,
                    trainable=False)
                tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
                with tf.Session() as sess:
                    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                          tokenization_info["do_lower_case"]])
            else:
                vocab_file = os.path.join(BertRanker.LOCAL_DIR, 'vocab.txt')
                do_lower_case = True

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)


    @staticmethod
    def preprocessdata(X, mode='split'):
        """
        Transform the dataset into Bert-edible features. If mode is 'train', then the data is split into
        train/test according to case_id, so that the cases in the test set are not represented in the train set.
        An over-samplind method is used to re-balance the datasets.
        Mode will indicate what to do with the data:
        prepare : just translate the input to features
        balance : balance the dataset, and translate to features
        split   : performs a train_test split to have a training dataset and an eval dataset.
                  In that case, the return value is a 2-uple. It is splitting based on cases, so the cases
                  in the eval dataset are not in the training dataset. Then the training dataset is relabanced and both
                   aretranslated to features.

        :param X: a Pandas DataFrame with columns 'case_id', 'case_text','candidate_text', 'candidate_is_noticed'
        :param mode: Which process to run.
        :return:  a 2-uple (train, test) of lists of bert.run_classifier.InputFeatures or a single list.
        """
        tokenizer = BertRanker.__create_tokenizer__()
        def __process__(df):
            df_InputExamples = df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                   text_a=x[BertRanker.DATA_COLUMN_A],
                                                                                   text_b=x[BertRanker.DATA_COLUMN_B],
                                                                                   label=x[BertRanker.LABEL_COLUMN]), axis=1)
            df_features = bert.run_classifier.convert_examples_to_features(
                df_InputExamples,
                BertRanker.LABEL_LIST,
                BertRanker.MAX_SEQ_LENGTH,
                tokenizer
            )
            return df_features

        data = X
        ros = RandomOverSampler()
        if mode == 'split':
            # Split so that cases in the test set are not in the training set
            train_cases, test_cases = train_test_split(data['case_id'].unique(), train_size=0.75)
            train = data[data['case_id'].isin(train_cases)]
            test = data[data['case_id'].isin(test_cases)]

            train_resampled, _ = ros.fit_resample(train, train[BertRanker.LABEL_COLUMN])
            train_df = pd.DataFrame(train_resampled, columns=data.columns.values)
            test_df = test

            return __process__(train_df), __process__(test_df)

        if mode == 'balance':
            data_resampled, _ = ros.fit_resample(data, data[BertRanker.LABEL_COLUMN])
            data_df = pd.DataFrame(data_resampled, columns=data.columns.values)
        else:
            data_df = data

        return __process__(data_df)

    def train_model(self, X=None, train_data=None, eval_data=None):
        """

        :param X: a dataframe with 3 columns : 'case_text', 'candidate_text', 'candidate_is_noticed', which is preprocessed,
                  or a 2-uple (train, test) of lists of bert.run_classifyer.InputFeatures
        :return:
        """
        # Convert our train and test features to InputFeatures that BERT understands.
        if X is not None:
            if isinstance(X, pd.DataFrame):
                train_features, eval_features, _ = self.preprocessdata(X, 'split')
            else:
                train_features, eval_features = X
        else:
            if isinstance(train_data, pd.DataFrame):
                train_features = self.preprocessdata(train_data, 'balance')
            else:
                train_features = train_data

            if isinstance(eval_data, pd.DataFrame):
                eval_features = self.preprocessdata(eval_data, 'prepare')
            else:
                eval_features = eval_data
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

        model_fn = BertRanker.__model_fn_builder__(
            num_labels=len(BertRanker.LABEL_LIST),
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
            seq_length=BertRanker.MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=False)

        # Create an input function for training. drop_remainder = True for using TPUs.
        eval_input_fn = bert.run_classifier.input_fn_builder(
            features=eval_features,
            seq_length=BertRanker.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, throttle_secs=60, start_delay_secs=0)

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
            data_features = self.preprocessdata(X, mode='prepare')
        else:
            data_features = X
        tf.logging.info('Finished with examples_to_features')

        # Compute train and warmup steps from batch size
        # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
        OUTPUT_DIR = self.parameters['--outputdir']
        BATCH_SIZE = int(self.parameters['--batchsize'])

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(model_dir=OUTPUT_DIR)

        model_fn = BertRanker.__model_fn_builder__(
            num_labels=len(BertRanker.LABEL_LIST),
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
            seq_length=BertRanker.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)

        checkpoint_file = None
        if checkpoint is not None:
            checkpoint_file = os.path.join(OUTPUT_DIR, 'model.ckpt-{}'.format(checkpoint))

        tf.logging.info('Starting with Predicting')
        predictions_iterator = estimator.predict(input_fn=predict_input_fn, checkpoint_path=checkpoint_file, yield_single_examples=True)
        predictions = []
        for prediction in tqdm(predictions_iterator, total=len(X)):
            predictions.append(prediction)

        scores = [p['probabilities'][1] for p in predictions]
        return scores


def main():
    args = docopt(__doc__, version='COLIEE v1.0')

    BertRanker.__configure_class_vars__(args)

    # Read data
    if args['--encode'] is True:
        input_file = args['--text']
        data = pd.read_csv(input_file)
        mode = 'balance' if args['--balance'] else 'prepare'
        inputs = BertRanker.preprocessdata(data, mode='balance')
        pickle.dump(inputs, open(args['--saveinputs'], 'wb'))
        sys.exit(0)

    if args['--train'] is True:
        br = BertRanker(parameters=args)
        if args['--inputs'] is not None or args['--text'] is not None:
            if args['--inputs'] is not None:
                train_data = pickle.load(open(args['--inputs'], 'rb'))
            else:
                train_data = pd.read_csv(args['--text'])
            br.train_model(X=train_data)
        else:
            if args['--traindata'].endswith('.csv'):
                train_data = pd.read_csv(args['--traindata'])
            else:
                train_data = pickle.load(open(args['--traindata'], 'rb'))

            if args['--evaldata'].endswith('.csv'):
                test_data = pd.read_csv(args['--evaldata'])
            else:
                test_data = pickle.load(open(args['--evaldata'], 'rb'))
            br.train_model(train_data=train_data, eval_data=test_data)

    if args['--rank'] is True:
        br = BertRanker(parameters=args)
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
