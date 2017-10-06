#!/usr/bin/env python

import sys
#import numpy as np

## Scikit-learn imports (for svm training and data manipulation)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn import svm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

## pytorch imports
from torch import Tensor
#from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torch.utils.data import TensorDataset,DataLoader
from torch.autograd import Variable
from torch.nn import SoftMarginLoss

import numpy as np

## All categories in the 20 newsgroups data.
cats = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

def main(args):
    scorer = make_scorer(my_scorer)
    vectorizer = Vectorizer()
    epochs = 50
    valid_pct = 0.2

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
            #cat_targets = to_categorical(binary_targets)
            #targets = newsgroups_train.target * 2 - 1
            train_X_tensor = Tensor(scaled_vectors.toarray())
            train_y_tensor = Tensor(class_targets)
            pyt_data = TensorDataset(train_X_tensor,  train_y_tensor)
            data_loader = DataLoader(pyt_data, batch_size=32, shuffle=True)

            (train_iter,) = data.Iterator.splits((pyt_data,), batch_size=32,
                device=-1, sort = False,  sort_key=lambda x: x.sum() )
            print("Classifying %s vs. %s" % (cat1, cat2))
            end_train_range = int((1-valid_pct) * vectors.shape[0])
            print("Training on first %f of data, %d instances" % (100*(1-valid_pct), end_train_range))

            ## Get NN performance
            print("Classifying with svm-like (no hidden layer) neural network:")
            svmlike_model = SvmlikeModel(vectors.shape[1], lr=0.01)
            iterations = 0
            for epoch in range(epochs):
            #     # batches:
            #     train_iter.init_epoch()
            #     for batch_idx, batch in enumerate(train_iter):
            #         svmlike_model.train()
            #         answer = svmlike_model(batch)
            #         loss = svmlike_model.criterion()(answer, batch.label)
            #         loss.backward()
            #         svmlike_model.update()
            #
            # single nistance at a time:
                epoch_loss = 0
                for item_ind in range(end_train_range):
                    item = pyt_data[item_ind]
                    svmlike_model.train();
                    iterations += 1
                    answer = svmlike_model(Variable(item[0]))
                    loss = svmlike_model.criterion(answer,  Variable(Tensor((item[1],))))
                    if np.isnan(loss.data[0]):
                        sys.stderr.write("Training example %d has nan loss" % (item_ind))
                    epoch_loss += loss
                    loss.backward();
                    svmlike_model.update()
                    #print("Epoch %d with loss %f and cumulative loss %f" % (epoch, loss.data[0], epoch_loss.data[0]))

                valid_batch = pyt_data[end_train_range:][0]
                valid_answer = svmlike_model(Variable(valid_batch))
                valid_loss = svmlike_model.criterion(valid_answer, Variable(pyt_data[end_train_range:][1]))
                valid_f1 = f1_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy())
                print("Epoch %d with training loss %f and validation loss %f and validation f1=%f" %
                    (epoch, epoch_loss.data[0], valid_loss.data[0], valid_f1))

            #score = np.average(cross_val_score(sp_model, vectors.toarray(), newsgroups_train.target, scoring=scorer, n_jobs=1, fit_params=dict(verbose=1, callbacks=[early_stopping])))
            #param_grid={'l2_weight':[0.001], 'lr':[0.1]}
            #clf = GridSearchCV(sp_model, param_grid, scoring=scorer)
            #clf.fit(vectors.toarray(), class_targets)
            #print("\nScore of nn cross-validation=%f with parameters=%s" % (clf.best_score_, clf.best_params_))

            ####################################################################
            # Below here is the actual svm
            ####################################################################
            print("Classifying with linear svm:")
            max_score = max_c = 0
            params = {'C':[0.01, 0.1, 1.0, 10.0, 100]}
            svc = svm.LinearSVC()
            clf = GridSearchCV(svc, params, scoring=scorer)
            clf.fit(vectors, binary_targets)
            print("Best SVM performance was %f with c=%f" % (clf.best_score_, clf.best_params_['C']))

            sys.exit(-1)

class SvmlikeModel(nn.Module):
    def __init__(self, input_dims, lr=0.1):
        super(SvmlikeModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.loss = SoftMarginLoss()

    def train(self):
        nn.Module.train(self)
        self.optimizer.zero_grad()

    def forward(self, batch):
        x = self.fc1(batch)
        return x

    def criterion(self, system, actual):
        return self.loss(system, actual)

    def update(self):
        self.optimizer.step()

def my_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred)

if __name__ == "__main__":
    main(sys.argv[1:])
