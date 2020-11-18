# Code for Legal Information Retrieval with Generalized Language Models (COLIEE 2019)

##Data

* Dataset from this [Link](https://drive.google.com/file/d/10wd_LJjGAFMfnP2I9nqKlrqw2m5p3SZn/view?usp=sharing)
* BERT checkpoint from this [Link](https://drive.google.com/file/d/1yd0QAnln7sZiPCSMh-k8UsTk-HJ9_CDM/view?usp=sharing)

## Environment

The `requirements.txt` and `env.yml` files have been lost...

This was developed in the first quarter of 2019, so the current version of the modules available through conda or PyPi
might not work as expected.

Used modules:
* tensorflow (probably 1.11 1.12, not 2.xx)
* tensorflow_hub
* pandas
* docopt
* sklearn
* imblearn
* numpy
* scipy
* gensim
* matplotlib
* nltk
* langdetect
* pytrec_eval
* cacheout

It is using the original `bert` github:
* `cd Task_01`
* `git clone git@github.com:google-research/bert.git`
* folder `bert` is a subfolder of `Task_01`

It is likely it was using Python 3.7

## Usage

### `task_01.py`
Will run all the baselines and the necessary vectorizers for these baselines (LSA, BoW, ...).

### `bert_ranking.py`
The hub for all operations with BERT.
See in the file the documentation for the command line.

* encode: from original text to InputFeatures (tokenized text) for BERT input. The encoded dataset can be used as input 
for the training, if not done, then the encoding of a text dataset will be done as a preprocessing step before
running the training. IN: csv file with proper columns, OUT: feature binary file

* train: do the training

 