#!/usr/bin/env python

import sys
from os import path
from os.path import join
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#import numpy as np
import time

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
import torch

import numpy as np
import csv

from tmlib.models import SimpleModel, SvmlikeModel, L2VectorLoss
from tmlib.data import YelpPolarityDataset

def my_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def main(args):
    epochs = 100
    valid_pct = 0.2
    default_lr = 0.01
    default_c = 1.0
    default_decay = 0.0
    hidden_nodes = 128
    batch_size = 512

    if len(args) < 1:
        sys.stderr.write("One required argument: <train directory>\n")
        sys.exit(-1)

    gpu = False
    if torch.cuda.is_available():
        print("CUDA found so processing will be done on the GPU")
        gpu = True

    examples = []
    labels = []
    sys.stderr.write("Loading data\n")
    with open(join(args[0], 'train.csv')) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            labels.append(int(row[0]))
            examples.append(row[1])

    sys.stderr.write("Transforming data into feature vectors\n")
    scorer = make_scorer(my_scorer)
    vectorizer = Vectorizer(max_features=10000)

    vectors = vectorizer.fit_transform(examples)
    all_y = np.array(labels).astype('float32')
    ## These labels are 1-2 by definition, relabel to +1/-1 for hinge loss:
    binary_targets = all_y - 1
    class_targets = ((all_y-1)*2) - 1

    sys.stderr.write("Rescaling data\n")
    scaler = StandardScaler(with_mean=False)
    scaler.fit(vectors)
    ## Doesn't seem to matter
    scaled_vectors = scaler.transform(vectors).astype('float32')
    num_valid_instances = min(1000, int(valid_pct * vectors.shape[0]))
    end_train_range = vectors.shape[0] - num_valid_instances

    ####################################################################
    # Here is the actual svm
    ####################################################################
    # print("Classifying with linear svm:")
    # max_score = max_c = 0
    # params = {'C':[0.001, 0.01, 0.1, 1.0]}
    # svc = svm.LinearSVC()
    # clf = GridSearchCV(svc, params, scoring=scorer)
    # clf.fit(vectors, binary_targets)
    # print("Best SVM performance was with c=%f" % (clf.best_params_['C']))
    # svc = svm.LinearSVC(C=clf.best_params_['C'])
    # svc.fit(scaled_vectors[:end_train_range], binary_targets[:end_train_range])
    # score = svc.score(scaled_vectors[end_train_range:], binary_targets[end_train_range:])
    # print("** SVM score with standard validation set is **%f**" % (score))

    ####################################################################
    # transform the data into pytorch format:
    ####################################################################
    yelp_train_data = YelpPolarityDataset(scaled_vectors[:end_train_range], class_targets[:end_train_range])
    train_loader = DataLoader(yelp_train_data, shuffle=True, batch_size=batch_size, num_workers=1)
    yelp_valid_data = YelpPolarityDataset(scaled_vectors[end_train_range:], class_targets[end_train_range:])
    valid_loader = DataLoader(yelp_valid_data)

    ## Get NN performance
    iterations = 0
    my_lr = default_lr

    model = SvmlikeModel(vectors.shape[1], lr=my_lr, c=default_c, decay=default_decay)
    if gpu:
        model.cuda()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start = time.time()
        for data in train_loader:
            inputs, labels = data

            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.train();
            answer = model(inputs)
            loss = model.criterion(answer, labels)
            epoch_loss += loss
            loss.backward()
            model.update()

        train_time = time.time() - epoch_start
        #print("Training during epoch %d took %fs\n" % (epoch, train_time))

        valid_acc = 0.0
        valid_loss = 0.0
        for data in valid_loader:
            inputs, labels = data

            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            valid_answer = model(inputs)
            valid_loss += model.criterion(valid_answer, labels)

            data_proportion = float(inputs.size()[0]) / num_valid_instances
            valid_acc += data_proportion * accuracy_score(np.sign(valid_answer.cpu().data.numpy()), labels.cpu().data.numpy())

        prev_valid_acc = valid_acc

        end_time = time.time()
        duration = end_time - epoch_start

        if epoch % 1 == 0: print("Epoch %d took %ds with training loss %f and validation loss %f, acc=%f" %
                            (epoch, duration, epoch_loss.data[0], valid_loss.data[0], prev_valid_acc))


if __name__ == "__main__":
    main(sys.argv[1:])
