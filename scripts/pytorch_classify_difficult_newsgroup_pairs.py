#!/usr/bin/env python

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
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
import torch

import numpy as np

from tmlib.models import SimpleModel, SvmlikeModel, L2VectorLoss

## All categories in the 20 newsgroups data.
cats = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
SVM_LIKE=0
ONE_LAYER=1
SVM_INIT=2
SVM_REG=3

def my_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def main(args):
    scorer = make_scorer(my_scorer)
    vectorizer = Vectorizer()
    epochs = 100
    valid_pct = 0.2
    default_lr = 0.1
    default_c = 0.1
    default_decay = 0.0
    hidden_nodes = 128
    batch_size = 64

    if torch.cuda.is_available():
        print("CUDA found so processing will be done on the GPU")

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
            ## Target vectors for the SVM:
            binary_targets = newsgroups_train.target
            ## Put targets in the range -1 to 1 instead of 0/1 for the nn hinge loss
            class_targets = newsgroups_train.target * 2 - 1

            train_X_tensor = Tensor(scaled_vectors.toarray())
            train_y_tensor = Tensor(class_targets)
            if torch.cuda.is_available():
                pyt_data = TensorDataset(train_X_tensor.cuda(),  train_y_tensor.cuda())
            else:
                pyt_data = TensorDataset(train_X_tensor,  train_y_tensor)


            print("Classifying %s vs. %s" % (cat1, cat2))
            end_train_range = int((1-valid_pct) * vectors.shape[0])

            ## Get NN performance
            iterations = 0
            train_X = pyt_data[:end_train_range][0]
            train_y = train_y_tensor[:end_train_range]
            valid_X = pyt_data[end_train_range:][0]
            valid_y = train_y_tensor[end_train_range:]

            my_lr = default_lr

            ####################################################################
            # Here is the actual svm
            ####################################################################
            print("Classifying with linear svm:")
            max_score = max_c = 0
            params = {'C':[0.01, 0.1, 1.0, 10.0, 100]}
            svc = svm.LinearSVC()
            clf = GridSearchCV(svc, params, scoring=scorer)
            clf.fit(vectors, binary_targets)
            print("Best SVM performance was with c=%f" % (clf.best_params_['C']))
            svc = svm.LinearSVC(C=clf.best_params_['C'])
            svc.fit(scaled_vectors[:end_train_range], binary_targets[:end_train_range])
            score = svc.score(scaled_vectors[end_train_range:], binary_targets[end_train_range:])
            print("** SVM score with standard validation set is **%f**" % (score))


            ####################################################################
            # Here are the different NN models
            ####################################################################
            for model_ind in (SVM_LIKE,):
                if model_ind == SVM_LIKE:
                    print("Classifying with svm-like (no hidden layer) neural network:")
                elif model_ind == ONE_LAYER:
                    print("Classifying with one hidden layer neural network with %d hidden nodes" % (hidden_nodes))
                elif model_ind == SVM_INIT:
                    print("Classifying with one hidden layer with %d hidden nodes initialized by previous system" % (hidden_nodes))
                elif model_ind == SVM_REG:
                    print("Classifying with one hidden layer with %d hidden nodes initialized and regularized by svm-like system." % (hidden_nodes))

                for try_num in range(5):
                    init_reg = False
                    weight_reg = L2VectorLoss()
                    if model_ind == SVM_LIKE:
                        model = SvmlikeModel(vectors.shape[1], lr=my_lr, c=default_c, decay=default_decay)
                        svmlike_model = model
                    elif model_ind == ONE_LAYER:
                        model = ExtendedModel(vectors.shape[1], hidden_nodes, lr=my_lr)
                    elif model_ind == SVM_INIT:
                        model = ExtendedModel(vectors.shape[1], hidden_nodes, lr=my_lr, init=saved_weights)
                    else:
                        model = ExtendedModel(vectors.shape[1], hidden_nodes, lr=my_lr, init=saved_weights)
                        init_reg = True
                        init_weight_reg = L2VectorLoss()

                    ## Move the model to the GPU:
                    if torch.cuda.is_available():
                        model.cuda()

                    valid_answer = model(Variable(valid_X)).cpu()
                    valid_acc = prev_valid_acc = accuracy_score(np.sign(valid_answer.data.numpy()), valid_y.numpy())
                    nan = False

                    for epoch in range(epochs):
                        #print("Epoch %d" % (epoch))
                    # single instance at a time:
                        nan = False
                        epoch_loss = 0
                        ## Shuffle data:
                        shuffle = torch.randperm(end_train_range)
                        train_X = train_X[shuffle]
                        train_y = train_y[shuffle]
                        for batch_ind in range(end_train_range // batch_size):
                            start_ind = batch_ind * batch_size
                            end_ind = min(start_ind+batch_size, end_train_range)
                            item = (train_X[start_ind:end_ind], train_y[start_ind:end_ind])
                            model.train();
                            iterations += 1
                            answer = model(Variable(item[0]))
                            if epoch == 0 and model_ind == SVM_INIT:
                                svm_answer = svmlike_model(Variable(item[0]))

                            loss = model.criterion(answer[:,0],  Variable(item[1]))
                            if np.isnan(loss.data[0]):
                                sys.stderr.write("Training batch %d at epoch %d has nan loss\n" % (batch_ind, epoch))
                                nan = True
                                break

                            epoch_loss += loss
                            loss.backward()

                            ## If we're on the model that regularizes towards the initial conditions then run
                            ## that loss function:
                            if init_reg and epoch > 0:
                                init_weight_reg_loss = init_weight_reg(model.fc1.weight[0,:], saved_weights)
                                init_weight_reg_loss.backward()

                            ## Always do L2 weight regularization:
                            #weight_reg_loss = weight_reg(model.fc1.weight, torch.zeros(model.fc1.weight.size()))
                            #weight_reg_loss.backward()

                            model.update()
                            #print("Epoch %d with loss %f and cumulative loss %f" % (epoch, loss.data[0], epoch_loss.data[0]))

                        if nan:
                            break

                        valid_batch = pyt_data[end_train_range:][0]
                        valid_answer = model(Variable(valid_batch))[:,0]
                        valid_loss = model.criterion(valid_answer, Variable(pyt_data[end_train_range:][1]))
                        #valid_f1 = f1_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy(), pos_label=-1)
                        valid_acc = accuracy_score(np.sign(valid_answer.cpu().data.numpy()), valid_y.numpy())
                        if epoch % 10 == 0: print("Epoch %d with training loss %f and validation loss %f, acc=%f" %
                            (epoch, epoch_loss.data[0], valid_loss.data[0], prev_valid_acc))
                        prev_valid_acc = valid_acc

                    if not nan:
                        print("** Finished with validation accuracy **%f**" % (valid_acc))
                        if model_ind == 0:
                            saved_weights = Variable(model.fc1.weight[0,:].data)
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



if __name__ == "__main__":
    main(sys.argv[1:])
