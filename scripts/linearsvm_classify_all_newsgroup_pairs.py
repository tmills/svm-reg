#!/usr/bin/env python

import sys
import numpy as np

## Scikit-learn imports (for svm training and data manipulation)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

## All categories in the 20 newsgroups data.
cats = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

def main(args):
    scorer = make_scorer(accuracy_score)
    vectorizer = Vectorizer()
    epochs = 50
    valid_pct = 0.2

    for cat1ind in range(len(cats)-1):
        cat1 = cats[cat1ind]
        for cat2ind in range(cat1ind+1,len(cats)):
            cat2 = cats[cat2ind]
            print("Classifying %s vs. %s" % (cat1, cat2))
            subcats = [cat1, cat2]
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=subcats)
            vectors = vectorizer.fit_transform(newsgroups_train.data)


            #vectors = vectors.toarray()
            scaler = StandardScaler(with_mean=False)
            scaler.fit(vectors)
            ## Doesn't seem to matter
            scaled_vectors = scaler.transform(vectors)
            ## Put targets in the range -1 to 1 instead of 0/1
            binary_targets = newsgroups_train.target
            class_targets = newsgroups_train.target * 2 - 1
            ####################################################################
            # Below here is the actual svm
            ####################################################################
            max_score = max_c = 0
            params = {'C':[0.01, 0.1, 1.0, 10.0, 100]}
            svc = svm.LinearSVC()
            clf = GridSearchCV(svc, params, scoring=scorer)
            clf.fit(vectors, binary_targets)
            print("Best SVM performance was acc=%f with c=%f" % (clf.best_score_, clf.best_params_['C']))


if __name__ == "__main__":
    main(sys.argv[1:])
