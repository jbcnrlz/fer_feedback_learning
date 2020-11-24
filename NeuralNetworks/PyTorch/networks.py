from torch import nn
import torch.nn.functional as F

class TimeSeriesLearning(nn.Module):
    def __init__(self,classes,in_channels=4):
        super(TimeSeriesLearning,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linearFeatures = nn.Sequential(
            nn.Linear(179776, 2048)
        )

        self.lstm = nn.LSTM(2048,1024)

        self.featureTimeLearning = nn.Linear(1024, classes)

    def forward(self,x):
        outfeatures = self.features(x)
        outfeatures = self.linearFeatures(outfeatures.view(len(x), -1))
        lstmout, _ = self.lstm(outfeatures.view(len(outfeatures),1,-1))
        tag_space = self.featureTimeLearning(lstmout.view(len(lstmout), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

