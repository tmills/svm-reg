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

## Keras imports (for neural network eval):
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras import regularizers
from keras.initializers import Zeros, Ones, RandomNormal, RandomUniform, TruncatedNormal, Orthogonal, Identity, lecun_uniform, glorot_normal, glorot_uniform, he_normal, lecun_normal, he_uniform
from keras import backend as K

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
            #vectors = vectors.toarray()
            scaler = StandardScaler(with_mean=False)
            print(scaler.fit(vectors))
            ## Doesn't seem to matter
            scaled_vectors = scaler.transform(vectors)
            ## Put targets in the range -1 to 1 instead of 0/1
            binary_targets = newsgroups_train.target
            class_targets = newsgroups_train.target * 2 - 1
            #targets = newsgroups_train.target * 2 - 1

            print("Classifying %s vs. %s" % (cat1, cat2))

            ## Get NN performance
            print("Classifying with svm-like (no hidden layer) neural network:")
            #model = get_model(vectors.shape[1])
            sp_model = KerasClassifier(build_fn=get_svmlike_model, input_dims=vectors.shape[1], l2_weight=0.01, epochs=50, validation_split=0.2)

            #score = np.average(cross_val_score(sp_model, vectors.toarray(), newsgroups_train.target, scoring=scorer, n_jobs=1, fit_params=dict(verbose=1, callbacks=[early_stopping])))
            param_grid={'l2_weight':[0.001,0.01], 'lr':[0.001,0.01]}
            clf = GridSearchCV(sp_model, param_grid, scoring=scorer)
            clf.fit(vectors.toarray(), class_targets)
            print("\nScore of nn cross-validation=%f with parameters=%s" % (clf.best_score_, clf.best_params_))

            print("Classifying with linear svm:")
            max_score = max_c = 0
            params = {'C':[0.01, 0.1, 1.0, 10.0, 100]}
            svc = svm.LinearSVC()
            clf = GridSearchCV(svc, params, scoring=scorer)
            clf.fit(vectors, targets)
            print("Best SVM performance was %f with c=%f" % (clf.best_score_, clf.best_params_['C']))

            sys.exit(-1)


def get_svmlike_model(input_dims, l2_weight=0.01, lr=0.1, initializer=glorot_uniform()):

    model = Sequential()
    ## Default initializer is glorot_uniform
    model.add(Dense(1, use_bias=True, input_dim=input_dims,
                kernel_initializer=initializer,
                kernel_regularizer=regularizers.l2(l2_weight),
                activation='linear'))
    #model = Model(input=inputs, outputs=predictions)``
    optimizer = SGD(lr=lr)
    model.compile(optimizer=optimizer,
                loss='hinge',
                metrics=['accuracy'])
    return model

def get_mlp_model(input_dims, nodes=64, l2_weight=0.01, lr=0.1):
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dims, activation='relu'))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(l2_weight), activation='linear'))
    optimizer = SGD(lr=lr)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def redef_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true/2. + 1, K.round(y_pred/2. + 1)), axis=-1)


if __name__ == "__main__":
    main(sys.argv[1:])
