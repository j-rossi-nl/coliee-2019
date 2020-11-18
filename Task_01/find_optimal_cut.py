import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

INPUT_FILE = 'ranking/v3_train_scores.txt'
WITH_QRELS = 'text_summarized_200.csv'


scores = pd.read_csv(INPUT_FILE).set_index(['case_id', 'candidate_id'])
original = pd.read_csv(WITH_QRELS).set_index(['case_id', 'candidate_id'])
mix = scores.join(original).reset_index()[['case_id', 'candidate_id', 'score', 'candidate_is_noticed']]

# Restrict to only the test cases
test_cases = pd.read_csv('test_cases_id.csv', names=['case_id'], header=None).set_index('case_id')
mix = mix.set_index('case_id').join(test_cases, how='inner').reset_index()

number_cases = len(mix['case_id'].unique())
number_candidates= 200

print('mix: {}'.format(mix.shape))
print('Number of cases: {}'.format(number_cases))

nb_rels = {}
for case_id, rel in mix[mix['candidate_is_noticed'] == True].groupby(by='case_id').count()['candidate_id'].iteritems():
    nb_rels[case_id] = rel


cuts = []
for cut in np.arange(0.9, 1.0, 0.0001):
    results = mix[mix['score'] > cut]

    # If empty dataframe (ie : there is no candidate with a higher score than the cut)
    if len(results) == 0:
        avg_r = 0
        avg_p = 0
        avg_f1 = 0
    else:
        nb_found = {}
        nb_returned = {}

        for case_id, found in results[results['candidate_is_noticed'] == True].groupby(by='case_id').count()['candidate_id'].iteritems():
            nb_found[case_id] = found
        for case_id, returned in results.groupby(by='case_id').count()['candidate_id'].iteritems():
            nb_returned[case_id] = returned

        recall = {}
        precision = {}
        f1 = {}

        for case_id, rels in nb_rels.items():
            found = nb_found[case_id] if case_id in nb_found else 0
            returned = nb_returned[case_id] if case_id in nb_returned else 0
            r = found / rels
            p = found / returned if returned > 0 else 0
            recall[case_id] = r
            precision[case_id] = p
            f1[case_id] = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        # MACRO Average
        #avg_r = sum([v for _,v in recall.items()]) / len(recall)
        #avg_p = sum([v for _,v in precision.items()]) / len(precision)
        #avg_f1 = sum([v for _,v in f1.items()]) / len(f1)

        # MICRO Average
        avg_r = sum([v for _,v in nb_found.items()]) / sum([v for _,v in nb_rels.items()])
        avg_p = sum([v for _,v in nb_found.items()]) / sum([v for _,v in nb_returned.items()])
        avg_f1 = (2 * avg_r * avg_p) / (avg_r + avg_p)

    #print('cut={:0.2f} r={:0.4f} p={:0.4f} f2={:0.4f}'.format(cut, avg_r, avg_p, avg_f1))
    cuts.append([cut, avg_r, avg_p, avg_f1])

cuts_df = pd.DataFrame(cuts, columns=['cut', 'r', 'p', 'f1'])
for metric in ['r', 'p', 'f1']:
    print('Best cut for {}: \n{}\n'.format(metric, cuts_df.sort_values(metric, ascending=False)[['cut', metric]][:1]))

# Best at F1
best_f1 = cuts_df.sort_values('f1', ascending=False)[['cut', metric]][:1]['cut'].values[0]
print('Values at best cut: R={:0.2f} P={:0.2f}'.format(cuts_df[cuts_df['cut'] == best_f1]['r'].values[0],
                                                       cuts_df[cuts_df['cut'] == best_f1]['p'].values[0]))