"""bert_tf_ranking
Usage:
bert_tf_ranking --train --traindata=FILE --evaldata=FILE [--outputdir=DIR --batchsize=INT --epochs=INT]
bert_tf_ranking --rank --testdata=FILE [--testref=CSV] [--submission=FILE --cut=VALUE] [--rawfile=FILE] [--outputdir=DIR --batchsize=INT ] [--checkpoint=CKP]

Options:
  -h                  Show this screen
  --traindata=FILE    FILE can be either a CSV or a LIBSVM, indicated by file extension
  --evaldata=FILE     FILE can be either a CSV or a LIBSVM, indicated by file extension
  --testdata=FILE     FILE can be either a CSV or a LIBSVM, indicated by file extension
  --testref=CSV       In case a LIBSVM was given, we don't have the query_id / candidate_id, give it in a CSV
  --outputdir=DIR     Folder for BERT checkpoints [default: bert_output]
  --batchsize=INT     Size of training batches  [default: 32]
  --epochs=INT        Number of training epochs [default: 5]
  --rank              Use a trained BERT model for ranking
  --cut=VALUE         The score value at which to cut
  --submission=FILE   The file to submit to the competition
  --rawfile=FILE      File in which to save the raw scores
  --checkpoint=CKP    Which checkpoint to use (CKP is a number)
"""
import tensorflow as tf
import os
import pandas as pd
import tensorflow_ranking as tfr
import numpy as np

from tensorflow.python.client import device_lib

from docopt import docopt
from bert_encoder import BertEncoder
from utils import libsvm_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'


class TFRanker:
    """

    """
    # In the TF-Ranking framework, a training instance is represented
    # by a Tensor that contains features from a list of documents
    # associated with a single query. For simplicity, we fix the shape
    # of these Tensors to a maximum list size and call it "list_size,"
    # the maximum number of documents per query in the dataset.
    # In this demo, we take the following approach:
    #   * If a query has fewer documents, its Tensor will be padded
    #     appropriately.
    #   * If a query has more documents, we shuffle its list of
    #     documents and trim the list down to the prescribed list_size.
    _LIST_SIZE=200

    # The total number of features per query-document pair.
    # We set this number to the number of features in the MSLR-Web30K
    # dataset.
    _NUM_FEATURES=768

    _HIDDEN_LAYER_DIMS = ["256", "16"]

    def __init__(self, parameters):
        # Store the paths to files containing training and test instances.
        # As noted above, we will assume the data is in the LibSVM format
        # and that the content of each file is sorted by query ID.
        self._DATA_PATH = {
            'train': parameters['--traindata'],
            'eval': parameters['--evaldata'],
            'test': parameters['--testdata']
        }
        # Parameters to the scoring function.
        self._BATCH_SIZE = int(parameters['--batchsize'])       #32

        # Define a loss function. To find a complete list of available
        # loss functions or to learn how to add your own custom function
        # please refer to the tensorflow_ranking.losses module.
        self._LOSS='pairwise_soft_zero_one_loss'

        self._MODEL_DIR = parameters['--outputdir']
        self.epochs = parameters['--epochs']


    """### Input Pipeline
    
    The first step to construct an input pipeline that reads your dataset and produces a `tensorflow.data.Dataset` object. In this example, we will invoke a LibSVM parser that is included in the `tensorflow_ranking.data` module to generate a `Dataset` from a given file.
    
    We parameterize this function by a `path` argument so that the function can be used to read both training and test data files.
    """

    def input_fn(self, mode):
        infile = self._DATA_PATH[mode]
        data_generator = None
        # Either it is a CSV file with the original text, and then we need to use the Bert Encoder
        if infile.endswith('.csv'):
            data = pd.read_csv(self._DATA_PATH[mode])
            data.sort_values(by=['case_id', 'candidate_id'], inplace=True, ascending=[True, True])
            data['embeddings'] = [x for x in BertEncoder.encode(data)]

            def data_generator_from_dataframe():
                for id in data['case_id'].unique():
                    chunk = data[data['case_id'] == id]
                    vecs = chunk['embeddings']
                    labels = chunk['candidate_is_noticed'].to_numpy()
                    matrix = np.expand_dims(np.vstack(vecs), axis=-1)
                    features_dict = {str(k+1): matrix[:,k] for k in range(TFRanker._NUM_FEATURES)}
                    yield features_dict, labels
            data_generator = data_generator_from_dataframe

        # Or it is a LIBSVM format file, and then we have a reader
        # Use the modified code, it was shuffling the input data !!!
        # Use text_2_libsvm.py for generation
        if infile.endswith('.libsvm'):
            data_generator = libsvm_generator(infile, TFRanker._NUM_FEATURES, TFRanker._LIST_SIZE)

        #
        # dataset = tf.data.Dataset.from_generator(
        #     data_generator,
        #     output_types=({str(k): tf.float32 for k in range(1,TFRanker._NUM_FEATURES+1)}, tf.float32),
        #     output_shapes=({str(k): tf.TensorShape([TFRanker._LIST_SIZE, 1]) for k in range(1,TFRanker._NUM_FEATURES+1)}, tf.TensorShape([TFRanker._LIST_SIZE]))
        # )

        # We don't have big datasets, let's load them once and for all
        gen = data_generator()
        all_data = [x for x in gen]
        X = {}
        for i in range(TFRanker._NUM_FEATURES):
            X[str(i + 1)] = np.stack([x[0][str(i + 1)] for x in all_data], axis=0)
        Y = np.stack([x[1] for x in all_data])

        dataset = tf.data.Dataset.from_tensor_slices((X, Y))

        if mode == 'train':
            dataset = dataset.shuffle(300).repeat().batch(self._BATCH_SIZE)
        else:
            dataset = dataset.batch(self._BATCH_SIZE)

        # Queue up a number of batches on the CPU side
        dataset = dataset.prefetch(8)

        # # Queue up batches asynchronously onto the GPU
        # # As long as there is a pool of batches CPU side a GPU prefetch of 1 is sufficient.
        # gpu = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        # if len(gpu) == 1:
        #     dataset.apply(tf.data.experimental.prefetch_to_device(gpu[0], buffer_size=8))

        #return dataset.make_one_shot_iterator().get_next()
        return dataset

    """### Scoring Function
    
    Next, we turn to the scoring function which is arguably at the heart of a TF Ranking model. The idea is to compute a relevance score for a (set of) query-document pair(s). The TF-Ranking model will use training data to learn this function.
    
    Here we formulate a scoring function using a feed forward network. The function takes the features of a single example (i.e., query-document pair) and produces a relevance score.
    """

    def example_feature_columns(self):
      """Returns the example feature columns."""
      feature_names = [
          "%d" % (i + 1) for i in range(0, TFRanker._NUM_FEATURES)
      ]
      return {
          name: tf.feature_column.numeric_column(
              name, shape=(1,), default_value=0.0) for name in feature_names
      }

    def make_score_fn(self):
        """Returns a scoring function to build `EstimatorSpec`."""

        def _score_fn(context_features, group_features, mode, params, config):
            """Defines the network to score a documents."""
            del params
            del config
            # Define input layer.
            example_input = [
                tf.layers.flatten(group_features[name])
                for name in sorted(self.example_feature_columns())
            ]
            input_layer = tf.concat(example_input, 1)

            cur_layer = input_layer
            for i, layer_width in enumerate(int(d) for d in TFRanker._HIDDEN_LAYER_DIMS):
                cur_layer = tf.layers.dense(
                    cur_layer,
                    units=layer_width,
                    activation="tanh")

            logits = tf.layers.dense(cur_layer, units=1)
            return logits

        return _score_fn

    """### Evaluation Metrics
    
    We have provided an implementation of popular Information Retrieval evalution metrics in the TF Ranking library.
    """

    def eval_metric_fns(self):
      """Returns a dict from name to metric functions.

      This can be customized as follows. Care must be taken when handling padded
      lists.

      def _auc(labels, predictions, features):
        is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
        clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
        clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
        return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
      metric_fns["auc"] = _auc

      Returns:
        A dict mapping from metric name to a metric function with above signature.
      """
      topns = [5, 10, 20]
      metric_fns = {}
      metric_fns.update({
          "metric/P@%d" % topn: tfr.metrics.make_ranking_metric_fn(
              tfr.metrics.RankingMetricKey.MRR, topn=topn)
          for topn in topns
      })
      metric_fns.update({
          "metric/OPACC@%d" % topn: tfr.metrics.make_ranking_metric_fn(
              tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY, topn=topn)
          for topn in topns
      })

      return metric_fns

    """### Putting It All Together
    
    We are now ready to put all of the components above together and create an `Estimator` that can be used to train and evaluate a model.
    """

    def get_estimator(self, hparams):
        """

        :param hparams:
        :return:
        """
        def _train_op_fn(loss):
            # Defines train op used in ranking head.
            return tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=hparams.learning_rate,
                optimizer="Adagrad")

        ranking_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(self._LOSS),
            eval_metric_fns=self.eval_metric_fns(),
            train_op_fn=_train_op_fn)

        run_config = tf.estimator.RunConfig(
            model_dir=self._MODEL_DIR,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            keep_checkpoint_max=5,
        )

        return tf.estimator.Estimator(
            model_fn=tfr.model.make_groupwise_ranking_fn(
                group_score_fn=self.make_score_fn(),
                group_size=1,
                transform_fn=None,
                ranking_head=ranking_head),
            params=hparams,
            config=run_config
        )

    """Let us instantiate and initialize the `Estimator` we defined above."""

    def train(self):
        hparams = tf.contrib.training.HParams(learning_rate=0.05)
        ranker = self.get_estimator(hparams)
        tf.estimator.train_and_evaluate(
            ranker,
            tf.estimator.TrainSpec(lambda: self.input_fn('train'), max_steps=100000),
            tf.estimator.EvalSpec(lambda: self.input_fn('eval'), throttle_secs=60, start_delay_secs=0)
        )

    def score(self):
        hparams = tf.contrib.training.HParams(learning_rate=0.05)
        ranker = self.get_estimator(hparams=hparams)
        scores = ranker.predict(input_fn=lambda: self.input_fn('test'))
        return scores


def main():
    args = docopt(__doc__, version='COLIEE v1.0')

    if args['--train'] is True:
        tr = TFRanker(parameters=args)
        tr.train()

    if args['--rank'] is True:
        tr = TFRanker(parameters=args)
        ref_file = args['--testref'] if args['--testref'] is not None else args['--testdata']
        test_data = pd.read_csv(ref_file).sort_values(by=['case_id', 'candidate_id'], ascending=[True, True])

        scores = tr.score()
        scores_list = [x for x in scores]

        # The score list is a list of NB_TEST_CASES elements, 1 per query
        # Each element is an array of 200 scores
        # They are the the pairwise scores for each query / candidate pair
        # We did the ordering by candidate_id ascending
        scores_unfolded_list = [y for x in scores_list for y in x]
        test_data['score'] = scores_unfolded_list

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
