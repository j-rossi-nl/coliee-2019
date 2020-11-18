"""

"""
import pandas as pd

INPUT_FILE = 'ranking/v3_train_scores.txt'
WITH_QRELS = 'text_summarized_200.csv'

def main():
    scores = pd.read_csv(INPUT_FILE).set_index(['case_id', 'candidate_id'])
    original = pd.read_csv(WITH_QRELS).set_index(['case_id', 'candidate_id'])
    full_mix = scores.join(original).reset_index()[['case_id', 'candidate_id', 'case_text', 'candidate_text', 'score', 'candidate_is_noticed']]
    high_scores = full_mix[full_mix['score'] > 0.9].drop(columns=['score'])

    # Restrict to only the test cases
    test_cases = pd.read_csv('test_cases_id.csv', names=['case_id'], header=None).set_index('case_id')
    high_test = high_scores.set_index('case_id').join(test_cases, how='inner').reset_index()
    high_train = high_scores[~high_scores['case_id'].isin(test_cases.reset_index()['case_id'])].reset_index()

    high_test.to_csv('HIGH_test.csv', index=False)
    high_train.to_csv('HIGH_train.csv', index=False)
    high_scores.to_csv('HIGH_all.csv', index=False)

    print('Train: {}   Test: {}'.format(high_train.shape, high_test.shape))
    print('Train:\n{}\nTest:\n{}'.format(high_train['candidate_is_noticed'].value_counts(), high_test['candidate_is_noticed'].value_counts()))

if __name__ == '__main__':
    main()