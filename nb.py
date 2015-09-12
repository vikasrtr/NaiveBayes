"""
Naive Bayes for SMS Spam classification

__author__: vikasrtr

"""

import string
import re
import math

# process a line


def process_line(l):

    # remove punctuation
    exclude = [',', '.', '!', '"', '\'', ':', '?', '&', '(', ')']
    l = ''.join([c for c in l if c not in exclude])

    # tokenize the words
    l = l.lower()
    words = re.split('\W+', l)

    # count words
    # use simple dictionary to ciunt
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0.0) + 1
    return counts


###########################################
# Load dataset and createa holdout set
with open('data/SMSSpamCollection') as f:
    lines = f.readlines()

# create a holdout set
holdout_percent = int(.9 * len(lines))
lines_test = lines[holdout_percent:]
lines = lines[0: holdout_percent]

# entire corpus
vocab = {}

# no. of spam and ham
priors = {
    'spam': 0.,
    'ham': 0.
}

# words in each category
cat_count = {
    'spam': {},
    'ham': {}
}

# process each line
for l in lines:
    if (l[0] == 'h'):
        category = 'ham'
        l = l[4:]
    else:
        category = 'spam'
        l = l[5:]

    # start counting
    priors[category] += 1

    # process each message
    word_counts = process_line(l)

    for word, count in word_counts.items():
        # check if word is to be added to vacab
        if word not in vocab:
            vocab[word] = 0.0

        # check if word in cat_count
        if word not in cat_count[category]:
            cat_count[category][word] = 0.0

        # update count for vab and cat_count
        vocab[word] += count
        cat_count[category][word] += count

# calculate prior probabilities
prior_spam = priors['spam'] / sum(priors.values())
prior_ham = priors['ham'] / sum(priors.values())

###########################################
# start prediction

# use logarithmic space for posterior
# predict for a line at a time

# correct categories
correct_cats = []
# predicted categories
preds = []

for l in lines_test:
    if (l[0] == 'h'):
        category = 'ham'
        l = l[4:]
    else:
        category = 'spam'
        l = l[5:]

    counts = process_line(l)

    log_prob_spam = 0.
    log_prob_ham = 0.

    for word, count in counts.items():
        # skip unknown words
        if word not in vocab:
            continue

        # calculate evidence
        p_word = vocab[word] / sum(vocab.values())

        # calculate likelihood
        p_word_given_spam = cat_count['spam'].get(
            word, 0.0) / sum(cat_count['spam'].values())
        p_word_given_ham = cat_count['ham'].get(
            word, 0.0) / sum(cat_count['ham'].values())

        # calculate posterior
        if p_word_given_spam > 0:
            log_prob_spam += math.log(count * p_word_given_spam / p_word)
        if p_word_given_ham > 0:
            log_prob_ham += math.log(count * p_word_given_ham / p_word)

    spam_score = math.exp(log_prob_spam + math.log(prior_spam))
    ham_score = math.exp(log_prob_ham + math.log(prior_ham))

    if spam_score > ham_score:
        pred = 'spam'
    else:
        pred = 'ham'

    # update categories
    correct_cats.append(category)
    preds.append(pred)

# stats
correct = 0
for i in range(len(preds)):
    if preds[i] == correct_cats[i]:
        correct += 1

print('Accuracy: {0}'.format(correct / len(preds)))

import numpy as np

y_pred = np.zeros(shape=(len(preds), 1))
y_test = np.zeros(shape=(len(preds), 1))

# convert predictions into 1/0 form for spam/ham
for i in range(len(preds)):
    if preds[i] == 'spam':
        y_pred[i] = 1
    if correct_cats[i] == 'spam':
        y_test[i] = 1

from sklearn import metrics
import pandas as pd
from ggplot import *

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
# ggplot(df, aes(x='fpr', y='tpr')) +\
#     geom_line() +\
#     geom_abline(linetype='dashed')

auc = metrics.auc(fpr, tpr)
ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
    geom_area(alpha=0.2) +\
    geom_line(aes(y='tpr')) +\
    ggtitle("ROC Curve w/ AUC=%s" % str(auc))
