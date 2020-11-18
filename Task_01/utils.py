# Copyright 2019 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input data parsing for ranking library.
Supports data stored in SequenceExample proto format.
SequenceExample (`tf.SequenceExample`) is defined in:
tensorflow/core/example/example.proto
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

# The document relevance label.
_LABEL_FEATURE = "label"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1.


def _libsvm_parse_line(libsvm_line):
    """Parses a single LibSVM line to a query ID and a feature dictionary.
    Args:
      libsvm_line: (string) input line in LibSVM format.
    Returns:
      A tuple of query ID and a dict mapping from feature ID (string) to value
      (float). "label" is a special feature ID that represents the relevance
      grade.
    """
    tokens = libsvm_line.split()
    qid = int(tokens[1].split(":")[1])

    features = {_LABEL_FEATURE: float(tokens[0])}
    key_values = [key_value.split(":") for key_value in tokens[2:]]
    features.update({key: float(value) for (key, value) in key_values})

    return qid, features


def _libsvm_generate(num_features, list_size, doc_list):
    """Unpacks a list of document features into `Tensor`s.
    Args:
      num_features: An integer representing the number of features per instance.
      list_size: Size of the document list per query.
      doc_list: A list of dictionaries (one per document) where each
        dictionary is a mapping from feature ID (string) to feature value (float).
    Returns:
      A tuple consisting of a dictionary (feature ID to `Tensor`s) and a label
      `Tensor`.
    """
    # Construct output variables.
    features = {}
    for fid in range(num_features):
        features[str(fid + 1)] = np.zeros([list_size, 1], dtype=np.float32)
    labels = np.ones([list_size], dtype=np.float32) * (_PADDING_LABEL)

    # Shuffle the document list and trim to a prescribed list_size.
    #TODO WHY WHY WHY the shuffle !!!!!!!
    #np.random.shuffle(doc_list)

    if len(doc_list) > list_size:
        doc_list = doc_list[:list_size]

    # Fill in the output Tensors with feature and label values.
    for idx, doc in enumerate(doc_list):
        for feature_id, value in six.iteritems(doc):
            if feature_id == _LABEL_FEATURE:
                labels[idx] = value
            else:
                features.get(feature_id)[idx, 0] = value

    return features, labels


def libsvm_generator(path, num_features, list_size, seed=None):
    """Parses a LibSVM-formatted input file and aggregates data points by qid.
    MODIF Juju : read the file at once
    Args:
      path: (string) path to dataset in the LibSVM format.
      num_features: An integer representing the number of features per instance.
      list_size: Size of the document list per query.
      seed: Randomization seed used when shuffling the document list.
    Returns:
      A generator function that can be passed to tf.data.Dataset.from_generator().
    """
    if seed is not None:
        np.random.seed(seed)

    doc_lists = []

    with open(path, "rt") as f:
        # cur indicates the current query ID.
        cur = -1
        docs = []
        for line in f:
            qid, doc = _libsvm_parse_line(line)
            if cur < 0:
                cur = qid

            # If qid is not new store the data and move onto the next line.
            if qid == cur:
                docs.append(doc)
                continue

            # Reset current pointer and re-initialize document list.
            cur = qid
            doc_lists.append(docs)
        doc_lists.append(docs)   # It is interrupted after the last line and 'continue'

    processed = [_libsvm_generate(num_features, list_size, x) for x in doc_lists]

    def inner_generator():
        """Produces a generator ready for tf.data.Dataset.from_generator.
        It is assumed that data points in a LibSVM-formatted input file are
        sorted by query ID before being presented to this function. This
        assumption simplifies the parsing and aggregation logic: We consume
        lines sequentially and accumulate query-document features until a
        new query ID is observed, at which point the accumulated data points
        are massaged into a tf.data.Dataset compatible representation.
        Yields:
          A tuple of feature and label `Tensor`s.
        """
        # A buffer where observed query-document features will be stored.
        # It is a list of dictionaries, one per query-document pair, where
        # each dictionary is a mapping from a feature ID to a feature value.
        for p in processed:
            yield p

    return inner_generator
