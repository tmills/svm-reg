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

## pytorch imports
from torch import Tensor
#from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torch.utils.data import TensorDataset,DataLoader
from torch.autograd import Variable
from torch.nn import MSELoss
import torch

cats = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']

def main(args):
    vectorizer = Vectorizer()
    epochs = 500
    valid_pct = 0.2
    default_lr = 10.0
    batch_size = 64
    MAX_TRIES = 5

    if torch.cuda.is_available():
      print("CUDA found so processing will be done on the GPU")

    for cat1ind in range(len(cats)-1):
        cat1 = cats[cat1ind]
        for cat2ind in range(cat1ind+1,len(cats)):
            cat2 = cats[cat2ind]
            subcats = [cat1, cat2]
            print("Classifying %s vs. %s" % (cat1, cat2))

            newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=subcats)
            vectors = vectorizer.fit_transform(newsgroups_train.data)

            #vectors = vectors.toarray()
            scaler = StandardScaler(with_mean=False)
            scaler.fit(vectors)
            ## Doesn't seem to matter
            scaled_vectors = scaler.transform(vectors)

            train_X_tensor = Tensor(scaled_vectors.toarray())
            if torch.cuda.is_available():
                pyt_data = TensorDataset(train_X_tensor.cuda(),  train_X_tensor.cuda())
            else:
                pyt_data = TensorDataset(train_X_tensor,  train_X_tensor)

            end_train_range = int((1-valid_pct) * vectors.shape[0])
            iterations = 0
            valid_X = pyt_data[end_train_range:][0]
            valid_y = train_X_tensor[end_train_range:]

            my_lr = default_lr

            model = AutoEncoderModel(vectors.shape[1], 2, lr=my_lr)

            for try_num in range(MAX_TRIES):
                nan = False
                for epoch in range(epochs):
                    nan = False
                    epoch_loss = 0
                    for batch_ind in range(end_train_range // batch_size):
                        start_ind = batch_ind * batch_size
                        end_ind = min(start_ind+batch_size, end_train_range)
                        item = pyt_data[start_ind:end_ind]
                        model.train();

                        iterations += 1
                        answer = model(Variable(item[0]))

                        loss = model.criterion(answer,  Variable(item[1]))
                        if np.isnan(loss.data[0]):
                            sys.stderr.write("Training batch %d at epoch %d has nan loss\n" % (batch_ind, epoch))
                            nan = True
                            break

                        epoch_loss += loss
                        loss.backward()
                        model.update()

                        if nan:
                            my_lr /= 2.
                            if try_num+1 < MAX_TRIES:
                                print("Attempting another try (%d) with learning rate halved to %f" % (try_num+1, my_lr))
                            else:
                                print("Every learning rate resulted in NaN. Quitting.")
                            break

                    ## Compute validation loss:
                    valid_batch = pyt_data[end_train_range:][0]
                    valid_answer = model(Variable(valid_batch))
                    valid_loss = model.criterion(valid_answer, Variable(pyt_data[end_train_range:][1]))
                    #valid_f1 = f1_score(np.sign(valid_answer.data.numpy()), pyt_data[end_train_range:][1].numpy(), pos_label=-1)
                    #valid_acc = accuracy_score(np.sign(valid_answer.cpu().data.numpy()), valid_y.numpy())
                    if epoch % 10 == 0: print("Epoch %d with training loss %f and validation loss %f" %
                        (epoch, epoch_loss.data[0], valid_loss.data[0]))

                    if nan:
                        break

                if not nan:
                    ## If we got through the epochs without nan then save the model
                    ## and don't take any more tries:

                    break

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

class AutoEncoderModel(SimpleModel):
    def __init__(self, input_dims, hidden_dims, lr=0.1):
        super(AutoEncoderModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dims, input_dims)

        self.loss = MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, batch):
        x = self.fc1(batch)
        x = self.relu(x)
        x = self.output(x)
        return x

if __name__ == "__main__":
    main(sys.argv[1:])
