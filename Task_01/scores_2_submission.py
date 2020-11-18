"""
Scores to Submission
Usage:
scores_2_submission --scores=CSV ( --scorecut=CUT | --rankcut=CUT ) --submission=FILE --run=NAME

Options:
  -h                  Show this screen
  --scores=CSV        Read scores from a CSV file
  --scorecut=CUT      Use a cut to select the top relevant
  --rankcut=CUT       Use a rank to select top relevant
  --submission=FILE   The submission file
  --run=NAME          The run name
"""
from docopt import docopt

import pandas as pd
import numpy as np

def main():
    args = docopt(__doc__, version='Score to Submission')
    scores = pd.read_csv(args['--scores'])
    nb_cases = len(scores['case_id'].unique())
    nb_candidates_per_case = int(len(scores) / nb_cases)
    assert len(scores) % nb_cases == 0

    if args['--scorecut'] is not None:
        retain = scores[scores['score'] > float(args['--scorecut'])]
    if args['--rankcut'] is not None:
        ranked = scores.sort_values(by=['case_id', 'score'], ascending=[True, False])
        ranked['rank'] = list(np.arange(1, nb_candidates_per_case+1, 1)) * nb_cases
        retain = ranked[ranked['rank'] <= int(args['--rankcut'])]

    with open(args['--submission'], 'w') as submission:
        run = args['--run']
        for i,v in retain.iterrows():
            case_id = int(v['case_id'])
            candidate_id = int(v['candidate_id'])
            submission.write('{:03d} {:03d} {}\n'.format(case_id, candidate_id, run))


if __name__ == '__main__':
    main()
