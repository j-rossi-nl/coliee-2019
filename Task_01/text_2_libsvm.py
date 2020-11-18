"""
Takes the train / test data and turn the txt file into libsvm files
The texts are encoded with BERT encoder: 1 set of features (768 dense features) for each pair query / candidate
The case_id is the used as query id
The pairs are given in ascending order of case_id and candidate_id
"""

import pandas as pd

from bert_encoder import BertEncoder

INPUT_TRAIN = 'data/text/train_summarized_200.csv'
INPUT_EVAL  = 'data/text/eval_summarized_200.csv'
INPUT_TEST = 'data/text/test_summarized_200.csv'

OUTPUT = {
    INPUT_TRAIN: 'data/libsvm/train_features.libsvm',
    INPUT_EVAL: 'data/libsvm/eval_features.libsvm',
    INPUT_TEST: 'data/libsvm/test_features.libsvm'
}

INPUTS = [INPUT_TEST]

for input in INPUTS:
    data = pd.read_csv(input)
    data.sort_values(by=['case_id', 'candidate_id'], inplace=True, ascending=[True, True])
    data['embeddings'] = [x for x in BertEncoder.encode(data)]

    with open(OUTPUT[input], 'w') as output:
        for _, sample in data.iterrows():
            vec = sample['embeddings']
            label = int(sample['candidate_is_noticed'])
            case_id = sample['case_id']
            candidate_id = sample['candidate_id']
            features = ' '.join(['{}:{}'.format(i+1,v) for i,v in enumerate(vec)])
            output.write('{} qid:{} {} cid:{}\n'.format(label, case_id, features, candidate_id))

