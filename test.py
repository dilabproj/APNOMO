import os
import numpy as np
import pandas as pd
import argparse
import torch
from model import ours
from dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from utils import *
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from early_stop import *
from sklearn.linear_model import LogisticRegression
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
start = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--moo", type=str, default="", help="tune moo: 1:A, 2:P, 3:R, 4:F, 5:E, 6:Ea, 7:ET, while 1=True & 0=False")
parser.add_argument("--k", type=int, default=10, help="subsequence number k")
parser.add_argument("--note", type=str, default="", help="other note")
parser.add_argument("--type", type=str, default="", help="With what situation")
parser.add_argument("--cuda", type=int, default=3, help="Which gpu do you want to use.")
parser.add_argument("--print_every", type=float, default=1, help="print the result on every N of epochs")

# Dataset hyperparameters
parser.add_argument("--test_threshold", type=float, default=0.5, help="over this probability will be considered an anomaly.")
parser.add_argument("--dataset", type=str, help="Dataset to load. Available: Synthetic")

# Model hyperparameters
parser.add_argument("--N_me", type=int, default=1, help="number of me")
parser.add_argument("--N_neighbor", type=int, default=1, help="number of the neighbor chosen from left and right")

parser.add_argument("--cut", type=float, default=1, help="how many cut to one time series. if -1, it means moving step = 1, this will add the memory pressure, so is not used in the last version.")
parser.add_argument("--nhid", type=int, default=40, help="Number of dimensions of the hidden state of Benefitter") #[16,32,64,128]
parser.add_argument("--nlayers", type=int, default=2, help="Number of layers for ECM's RNN.") #[1,2,3]
parser.add_argument("--rnn_cell", type=str, default="GRU", help="Type of RNN to use in Benefitter. Available: GRU, LSTM")


# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.") #10
parser.add_argument("--nepochs", type=int, default=3, help="Number of epochs.") #50
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.") #0.001
parser.add_argument("--save", type=int, default=1, help="If saving the result into csv file.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.start = start
np.random.seed(args.random_seed)
print(args.dataset+args.type+"_subX.XX"+args.note)
print(args.dataset)
print(device)
print("Over-sampling with Me = {}, Neighbor = {}.".format(args.N_me,args.N_neighbor))

if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_save_path = args.model_save_path

    #load data
    train_loader, test_loader, data, test_data = load_data(args, without_shuffle=True)

    model = ours(ninp=test_data.N_FEATURES, T=test_data.ntimesteps, nclass=test_data.N_CLASSES, args=args).to(device)
    model.load_state_dict(torch.load(model_save_path+args.dataset+args.type+args.note+".pt"), strict=False)
    rates = [ str(float(i+1)/float(args.k)) for i in range(args.k)]

    for iii, rate in enumerate(rates):
        T=test_data.ntimesteps
        print("===========Now loading with earliness "+rate)
        model.eval()
        with torch.no_grad():
            for i, (X, y, ix) in enumerate(train_loader):
                X = torch.transpose(X[:,-1,:int(float(rate)*T)], 0, 1) #N x t/cut x t x ndim
                # --- Forward pass ---
                prob = model(X, feature=True)
                prob = prob.cpu().detach().numpy() #B x nhid
                if i == 0:
                    train_prob = prob
                    train_y_true = y.detach().numpy()
                else:
                    train_prob = np.concatenate((train_prob,prob), 0) #N x nhid
                    train_y_true = np.concatenate((train_y_true, y)) #N

        with torch.no_grad():
            for i, (X, y, ix) in enumerate(test_loader):
                X = torch.transpose(X[:,-1,:int(float(rate)*T)], 0, 1)
                # --- Forward pass ---
                prob = model(X, feature=True)
                prob = prob.cpu().detach().numpy()
                if i == 0:
                    test_prob = prob
                    y_true = y.detach().numpy()
                else:
                    test_prob = np.concatenate((test_prob,prob), 0) #N
                    y_true = np.concatenate((y_true, y)) #N

        ### Save train_prob train_y and test_prob test_y first.
        if iii == 0:
            train_probs = np.expand_dims( train_prob, axis=0 )# 1 x N
            test_probs = np.expand_dims( test_prob, axis=0 )# 1 x N
        else:
            train_probs = np.concatenate((train_probs, np.expand_dims( train_prob, axis=0 )),0) #set x N x nhid
            test_probs = np.concatenate((test_probs, np.expand_dims( test_prob, axis=0 )),0) #set x N x nhid
        train_y_true_from_rnn = train_y_true

        test_y_true = y_true
        # print(train_y_true_from_rnn)
        # print(y_true)


    for iii, rate in enumerate(rates):
        train_prob, train_y_true = neighbor_over_sampling(train_probs, train_y_true_from_rnn, idx=iii, 
                            args=args, N_neighbor=args.N_neighbor, N_me=args.N_me)
        
        model_lr = LogisticRegression(penalty='l2',solver='liblinear', random_state=args.random_seed)
        model_lr.fit(train_prob, train_y_true)

        train_lr_prob = model_lr.predict_proba(train_probs[iii]) #N x nclass
        train_lr_prob = train_lr_prob[:,-1]

        test_lr_prob = model_lr.predict_proba(test_probs[iii]) #N x nclass
        test_lr_prob = test_lr_prob[:,-1]

        if iii == 0:
            train_threshold_predictions = train_lr_prob.reshape(1, -1) # 1 x N
            threshold_predictions = test_lr_prob.reshape(1, -1) # 1 x N
        else:
            train_threshold_predictions = np.concatenate((train_threshold_predictions,train_lr_prob.reshape(1, -1)),0) #set x N
            threshold_predictions = np.concatenate((threshold_predictions,test_lr_prob.reshape(1, -1)),0) #set x N
            
    real_train_y_true = train_y_true_from_rnn

    ##count performance at each earliness:
    threshold = 0.5
    print("="*40+" Training earliness")
    count_each_prob(train_threshold_predictions.transpose(1,0), real_train_y_true, threshold)
    print("="*40+" Testing earliness")
    count_each_prob(threshold_predictions.transpose(1,0), test_y_true, threshold)

    threshold = early_stop_prob_dynamic(prediction = threshold_predictions, label = test_y_true, args = args,
                          train_prediction=train_threshold_predictions, train_label=real_train_y_true)

    ####Show the result with best threshold
    print("Show the result with best threshold {}".format(threshold))
    print("="*40+" Training earliness")
    count_each_prob(train_threshold_predictions.transpose(1,0), real_train_y_true, threshold)
    print("="*40+" Testing earliness")
    count_each_prob(threshold_predictions.transpose(1,0), test_y_true, threshold)
