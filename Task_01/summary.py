"""Summary
From the full text of the cases, create a dataset with summaries of cases

Usage:
summary --in=CSV --out=CSV

Options:
  -h                  Show this screen
  --in=CSV            File with full text
  --out=CSV           File with summarized text
"""

from docopt import docopt
from basics import ColieeData, ColieePreprocessor


def main():
    args = docopt(__doc__, version='COLIEE v1.0')

    INPUT_FILE = args['--in']
    OUTPUT_FILE = args['--out']

    d = ColieeData(INPUT_FILE)
    print('{} Cases, {} Candidates per case'.format(d.nb_cases, d.nb_candidates_per_case))
    d.apply_preprocessing(ColieePreprocessor.summarize)
    d.data.to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()