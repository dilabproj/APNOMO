import torch
from torch import nn
import numpy as np
from tqdm import tqdm

class ours(nn.Module):
    """
    Parameters
    ----------
    ninp : int
        number of features in the input data.
    nclasses : int
        number of classes in the input labels.
    nhid : int
        number of dimensions in the RNN's hidden states.
    rnn_cell : str
        which RNN memory cell to use: {LSTM, GRU, RNN}.
        (if defining your own, leave this alone)
    nlayers : int
        number of layers in the RNN.

    """
    def __init__(self, ninp, T, nclass, args): #nhid=50, rnn_cell="LSTM", nlayers=1):
        super(ours, self).__init__()

        self.nclass = nclass
        self.device = args.device
        self.cut = args.cut

        # --- Hyperparameters ---
        ninp = ninp

        self.rnn_cell = args.rnn_cell
        self.nhid = args.nhid
        self.nlayers = args.nlayers

        if self.cut == -1:
            self.move_steps = 1
        else:
            self.move_steps = T/self.cut

        # --- Sub-networks ---
        if self.rnn_cell == "LSTM":
            self.RNN = torch.nn.LSTM(ninp, self.nhid, self.nlayers)
        else:
            self.RNN = torch.nn.GRU(ninp, self.nhid, self.nlayers)

        self.out = torch.nn.Linear(self.nhid, self.nhid)
        self.out2 = torch.nn.Linear(self.nhid, self.nclass)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)


    def initHidden(self, bsz):
        """Initialize hidden states"""
        if self.rnn_cell == "LSTM":
            return (torch.zeros(self.nlayers, bsz, self.nhid, device= self.device),
                    torch.zeros(self.nlayers, bsz, self.nhid, device= self.device))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid, device= self.device)

    def forward(self, X, epoch=0, test=False, softmax_in_test=True, feature=False):
        X = X.to(self.device)
        T, B, V = X.shape # Assume input is of shape (TIMESTEPS x BATCH x VARIABLES)
        hidden = self.initHidden(X.shape[1])

        output, hidden = self.RNN(X, hidden) # T x N x nhid, 2 x N x nhid
        after_tanh = self.tanh(output[-1])
        if feature:
            return after_tanh

        logit = self.dropout(self.relu(self.out( after_tanh )))
        logit = self.out2( logit )
        
        if test:
            logit = self.softmax(logit)
            if softmax_in_test:
                return torch.argmax(logit, dim=1) #N
            else:
                return logit

        return logit #N x nclass


    def computeLoss(self, logits, y):
        logits = logits.to(self.device)
        y = y.to(self.device).to(torch.long)
        Loss = nn.CrossEntropyLoss()
        return Loss(logits, y)
