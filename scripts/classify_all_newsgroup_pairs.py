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

## Keras imports (for neural network eval):
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

## All categories in the 20 newsgroups data.
cats = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

def main(args):
    scorer = make_scorer(f1_score)
    vectorizer = Vectorizer()
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    for cat1ind in range(len(cats)-1):
        cat1 = cats[cat1ind]
        for cat2ind in range(cat1ind+1,len(cats)):
            cat2 = cats[cat2ind]
            subcats = [cat1, cat2]
            newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=subcats)
            vectors = vectorizer.fit_transform(newsgroups_train.data)

            print("Classifying %s vs. %s" % (cat1, cat2))

            ## Get NN performance
            print("Classifying with neural network:")
            #model = get_model(vectors.shape[1])
            sp_model = KerasClassifier(build_fn=get_model, input_dims=vectors.shape[1], nodes=128, epochs=50, validation_split=0.2)

            #history = model.fit(vectors.toarray(), newsgroups_train.target, epochs=10, validation_split=0.2)
            #print("History is %s" % (history))
            score = np.average(cross_val_score(sp_model, vectors.toarray(), newsgroups_train.target, scoring=scorer, n_jobs=1, fit_params=dict(verbose=1, callbacks=[early_stopping])))
            #param_grid=dict(nodes=[32,64,128], epochs=[5,10])
            #grid = GridSearchCV(estimator=sp_model, param_grid=param_grid)
            #grid_result = grid.fit(vectors.toarray(), newsgroups_train.target)
            print("\nScore of nn cross-validation=%f" % score)

            print("Classifying with linear svm:")
            max_score = max_c = 0
            for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
                score = np.average(cross_val_score(svm.LinearSVC(C=C), vectors, newsgroups_train.target, scoring=scorer, n_jobs=1))
                if score > max_score:
                    max_score = score
                    max_c = C

            print("Best SVM performance was %f with c=%f" % (max_score, max_c))


def get_model(input_dims, nodes=64):
    #inputs = Input(shape=(input_dims,))

    model = Sequential()
    #model.add(Dense(nodes, activation='relu', input_dim=input_dims))
    model.add(Dense(1, input_dim=input_dims, activation='sigmoid'))
    #model = Model(input=inputs, outputs=predictions)
    optimizer = SGD(lr=0.1)
    model.compile(optimizer=optimizer,
                loss='squared_hinge',
                metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main(sys.argv[1:])
