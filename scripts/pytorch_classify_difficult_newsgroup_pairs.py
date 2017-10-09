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
cats = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']

def main(args):
    scorer = make_scorer(my_scorer)
    vectorizer = Vectorizer()
    epochs = 50
    valid_pct = 0.2
    default_lr = 0.01
    hidden_nodes = 32

    for cat1ind in range(len(cats)-1):
        cat1 = cats[cat1ind]
        for cat2ind in range(cat1ind+1,len(cats)):
            cat2 = cats[cat2ind]
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
            #print("Training on first %f of data, %d instances" % (100*(1-valid_pct), end_train_range))

            ## Get NN performance
            iterations = 0
            valid_batch = pyt_data[end_train_range:][0]

            my_lr = default_lr

            for model_ind in range(3):
                if model_ind == 0:
                    print("Classifying with svm-like (no hidden layer) neural network:")
                elif model_ind == 1:
                    print("Classifying with one hidden layer neural network with %d hidden nodes" % (hidden_nodes))
                else:
                    print("Classifying with one hidden layer with %d hidden nodes initialized by previous system")

                for try_num in range(5):
                    if model_ind == 0:
                        model = SvmlikeModel(vectors.shape[1], lr=my_lr)
                    elif model_ind == 1:
                        model = ExtendedModel(vectors.shape[1], hidden_nodes, lr=my_lr)
                    elif model_ind == 2:
                        model = ExtendedModel(vectors.shape[1], hidden_nodes, lr=my_lr, init=saved_weights)

                    valid_answer = model(Variable(valid_batch))
                    valid_acc = prev_valid_acc = accuracy_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy())
                    nan = False

                    for epoch in range(epochs):
                    # single instance at a time:
                        nan = False
                        epoch_loss = 0
                        for item_ind in range(end_train_range):
                            item = pyt_data[item_ind]
                            model.train();
                            iterations += 1
                            answer = model(Variable(item[0]))
                            loss = model.criterion(answer,  Variable(Tensor((item[1],))))
                            if np.isnan(loss.data[0]):
                                sys.stderr.write("Training example %d at epoch %d has nan loss\n" % (item_ind, epoch))
                                nan = True
                                break

                            epoch_loss += loss
                            loss.backward();
                            model.update()
                            #print("Epoch %d with loss %f and cumulative loss %f" % (epoch, loss.data[0], epoch_loss.data[0]))

                        if nan:
                            break

                        valid_batch = pyt_data[end_train_range:][0]
                        valid_answer = model(Variable(valid_batch))
                        valid_loss = model.criterion(valid_answer, Variable(pyt_data[end_train_range:][1]))
                        valid_f1 = f1_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy(), pos_label=-1)
                        valid_acc = accuracy_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy())
                        #print("Epoch %d with training loss %f and validation loss %f, f1=%f, acc=%f" %
                        #    (epoch, epoch_loss.data[0], valid_loss.data[0], valid_f1, prev_valid_acc))
                        prev_valid_acc = valid_acc

                    if not nan:
                        print("Finished with validation accuracy %f" % (valid_acc))
                        if model_ind == 0:
                            saved_weights = model.fc1.weight
                        break
                    elif try_num+1 < 5:
                        my_lr /= 2.
                        print("Attempting another try (%d) with learning rate halved to %f" % (try_num+1, my_lr))
                    else:
                        print("Ran out of tries, giving up on this classification task.")


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
            print("Best SVM performance was acc=%f with c=%f" % (clf.best_score_, clf.best_params_['C']))

            #sys.exit(-1)

class SimpleModel(nn.Module):
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

class SvmlikeModel(SimpleModel):
    def __init__(self, input_dims, lr=0.1):
        super(SvmlikeModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1)
        self.loss = SoftMarginLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, batch):
        x = self.fc1(batch)
        return x

class ExtendedModel(SimpleModel):
    def __init__(self, input_dims, hidden_dims, lr=0.1, init=None):
        super(ExtendedModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        if not init is None:
            self.fc1.weight[0,:].data = init.data
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dims, 1)
        self.loss = SoftMarginLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, batch):
        x = self.fc1(batch)
        x = self.relu(x)
        x = self.output(x)
        return x

def my_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    main(sys.argv[1:])
