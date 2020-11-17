from torch import nn
import torch.nn.functional as F

class TimeSeriesLearning(nn.Module):
    def __init__(self,classes,hidden_dim,in_channels=4):
        super(TimeSeriesLearning,self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_channels,self.hidden_dim)

        self.featureTimeLearning = nn.Linear(hidden_dim, classes)

    def forward(self,x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        tag_space = self.featureTimeLearning(lstm_out.view(len(x), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores