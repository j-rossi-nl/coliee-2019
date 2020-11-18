"""bert_prepare_texts_pretraining
Usage:
bert_prepare_texts_pretraining --text=CSV --output=CSV

Options:
  -h                  Show this screen
  --text=CSV          Read texts from a CSV file, will require preprocessing
  --output=CSV       Output file
"""
import pandas as pd
import re
import numpy as np

from langdetect import detect as lang_detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import sent_tokenize
from docopt import docopt
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def language_detect(s):
    try:
        return lang_detect(s)
    except LangDetectException:
        return 'non-en'


def parallelize(data, func, num_of_processes):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func)


def parallelize_on_rows(data, func, num_of_processes):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def prepare_text(text):
    """
    Clean and Split into sentences

    :param text:
    :return:
    """
    clean = text.replace('\n', ' ')
    clean = re.sub(r'[ ]{2,}', ' ', clean)

    sentences = [s for s in sent_tokenize(clean) if len(s.split()) > 5]
    only_en = [s for s in sentences if language_detect(s) == 'en']

    return only_en



def main():
    args = docopt(__doc__, version='COLIEE v1.0')

    original_texts = pd.read_csv(args['--text'], names=['text'])

    sentences = [x for x in tqdm(parallelize_on_rows(original_texts['text'], prepare_text, 16), total=len(original_texts))]
    with open(args['--output'], 'w') as output:
        for doc in sentences:
            # Sentences from 1 document in separated lines
            for sentence in doc:
                output.write('{}\n'.format(sentence))
            # documents separated by empty line
            output.write('\n')

if __name__ == '__main__':
    main()