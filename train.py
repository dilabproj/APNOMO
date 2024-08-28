import os
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import argparse
import torch
from model import ours
from dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--type", type=str, default="", help="With what situation")
parser.add_argument("--cuda", type=int, default=3, help="Which gpu do you want to use.")
parser.add_argument("--print_every", type=float, default=1, help="print the result on every N of epochs")
parser.add_argument("--cut", type=float, default=1, help="how many cut to one time series. if -1, it means moving step = 1, this will add the memory pressure, so is not used in the last version.")

# Dataset hyperparameters
parser.add_argument("--test_threshold", type=float, default=0.5, help="over this probability will be considered an anomaly.")
parser.add_argument("--dataset", type=str, help="Dataset to load. Available: Synthetic")

# Model hyperparameters
parser.add_argument("--nhid", type=int, default=40, help="Number of dimensions of the hidden state of Benefitter") #[16,32,64,128]
parser.add_argument("--nlayers", type=int, default=2, help="Number of layers for ECM's RNN.") #[1,2,3]
parser.add_argument("--rnn_cell", type=str, default="GRU", help="Type of RNN to use in Benefitter. Available: GRU, LSTM")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.") #10
parser.add_argument("--nepochs", type=int, default=3, help="Number of epochs.") #50
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.") #0.001
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")

args = parser.parse_args()

## Setting device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
print(device)

def train(args, model, optimizer, scheduler, train_loader, test_loader, data, test_data):
    # --- training ---
    list_result = []
    training_loss = []
    training_locations = []
    training_predictions = []
    for epoch in tqdm( range(args.nepochs) ):
        loss_sum = 0
        model.train()
        for i, (X, y, ix) in enumerate(train_loader) :
            X = torch.transpose(X[:,-1], 0, 1)
            # --- Forward pass ---
            logits = model(X, epoch)
            # print(logits.cpu().detach().numpy().shape)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()

            loss = model.computeLoss(logits, y)

            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

            # if (i+1) % 10 == 0:
            #     print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.nepochs, i+1, len(train_loader), loss.item()))

        training_loss.append(np.round(loss_sum/len(train_loader), 3))
        scheduler.step()

        ### Plot result
        if (epoch+1) % args.print_every == 0:
            print("===="*20)
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.nepochs, loss.item()))
            loss_item = loss.item()
            model.eval()
            
            # --- Run model on train data ---
            with torch.no_grad():
                for i, (X, y, ix) in enumerate(train_loader):
                    X = torch.transpose(X[:,-1], 0, 1)
                    # --- Forward pass ---
                    preds = model(X, test=True)
                    preds = preds.cpu().detach().numpy() #N x nclass
                    if i == 0:
                        train_predictions = preds
                        y_true = y.detach().numpy()
                    else:
                        train_predictions = np.concatenate((train_predictions,preds)) #N
                        y_true = np.concatenate((y_true, y)) #N

            train_accuracy = np.round(accuracy_score(y_true, train_predictions), 4)
            print("Train Accuracy: {}".format(train_accuracy))
            print(classification_report(y_true, train_predictions,digits=4))
            train_result = classification_report(y_true, train_predictions, labels=[1, 0], target_names=['True', 'False'], output_dict=True)
            train_precision = train_result['True']['precision']
            train_recall = train_result['True']['recall']
            train_f1score = train_result['True']['f1-score']

            save_dict = {"loss":loss_item,"train_accuracy":train_accuracy,"train_precision":train_precision,"train_recall":train_recall,"train_f1score":train_f1score}

            # --- Run model on test data ---
            with torch.no_grad():
                for i, (X, y, ix) in enumerate(test_loader):
                    X = torch.transpose(X[:,-1], 0, 1)
                    # --- Forward pass ---
                    preds = model(X, test=True, softmax_in_test=False)
                    preds = preds.cpu().detach().numpy() #N x nclass
                    preds = preds[:,1].reshape(-1) # N
                    temp = np.zeros(len(preds))
                    temp[np.argwhere(preds>=args.test_threshold)] = 1
                    preds = temp
                    if i == 0:
                        test_predictions = preds
                        y_true = y.detach().numpy()
                    else:
                        test_predictions = np.concatenate((test_predictions,preds)) #N
                        y_true = np.concatenate((y_true, y)) #N

            test_accuracy = np.round(accuracy_score(y_true, test_predictions), 4)
            print("test Accuracy: {}".format(test_accuracy))
            print(classification_report(y_true, test_predictions,digits=4))
            test_result = classification_report(y_true, test_predictions, labels=[1, 0], target_names=['True', 'False'], output_dict=True)
            test_precision = test_result['True']['precision']
            test_recall = test_result['True']['recall']
            test_f1score = test_result['True']['f1-score']

            extra_dict = {"test_accuracy":test_accuracy,"test_precision":test_precision,"test_recall":test_recall,"test_f1score":test_f1score}
            save_dict.update(extra_dict)
            list_result.append(save_dict)

    dataframe = pd.DataFrame( list_result )
    if not exists( "result_train" ):
        os.mkdir( "result_train" )
    dataframe.to_csv('result_train/{}{}_nhid{}_nepochs{}.csv'.format(args.dataset,args.type,args.nhid,args.nepochs), mode='w', index=False)

    return model


def test(args, model, train_loader, test_loader, data, test_data):
    print("===="*50)
    print ("====== Now test the result...")
    model.eval()
    # --- Run model on train data ---
    with torch.no_grad():
        for i, (X, y, ix) in enumerate(train_loader):
            X = torch.transpose(X[:,-1], 0, 1)
            # --- Forward pass ---
            preds = model(X, test=True)
            preds = preds.cpu().detach().numpy() #N x nclass
            if i == 0:
                train_predictions = preds
                y_true = y.detach().numpy()
            else:
                train_predictions = np.concatenate((train_predictions,preds)) #N
                y_true = np.concatenate((y_true, y)) #N

    train_accuracy = np.round(accuracy_score(y_true, train_predictions), 4)
    print("Train Accuracy: {}".format(train_accuracy))
    print(classification_report(y_true, train_predictions,digits=4))
    train_result = classification_report(y_true, train_predictions, labels=[1, 0], target_names=['True', 'False'], output_dict=True)
    train_precision = train_result['True']['precision']
    train_recall = train_result['True']['recall']
    train_f1score = train_result['True']['f1-score']

    # --- Run model on test data ---
    with torch.no_grad():
        for i, (X, y, ix) in enumerate(test_loader):
            X = torch.transpose(X[:,-1], 0, 1)
            # --- Forward pass ---
            preds = model(X, test=True, softmax_in_test=False)
            preds = preds.cpu().detach().numpy() #N x nclass
            preds = preds[:,1].reshape(-1) # N
            temp = np.zeros(len(preds))
            temp[np.argwhere(preds>=args.test_threshold)] = 1
            preds = temp
            if i == 0:
                test_predictions = preds
                y_true = y.detach().numpy()
            else:
                test_predictions = np.concatenate((test_predictions,preds)) #N
                y_true = np.concatenate((y_true, y)) #N

    test_accuracy = np.round(accuracy_score(y_true, test_predictions), 4)
    print("test Accuracy: {}".format(test_accuracy))
    print(classification_report(y_true, test_predictions,digits=4))
    test_result = classification_report(y_true, test_predictions, labels=[1, 0], target_names=['True', 'False'], output_dict=True)
    test_precision = test_result['True']['precision']
    test_recall = test_result['True']['recall']
    test_f1score = test_result['True']['f1-score']

    return 


if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_save_path = args.model_save_path
    utils.makedirs(model_save_path)

    #load data
    train_loader, test_loader, data, test_data = load_data(args)

    #create model
    model = ours(ninp=data.N_FEATURES, T=data.ntimesteps, nclass=data.N_CLASSES, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    model = train(args, model, optimizer, scheduler, train_loader, test_loader, data, test_data)

    test(args, model, train_loader, test_loader, data, test_data)

    torch.save(model.state_dict(), model_save_path+args.dataset+str(args.type)+".pt")