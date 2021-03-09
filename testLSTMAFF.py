import torch, copy, time, numpy as np
from DatasetClasses.FEDatasets import AFFData, AFFDataBlock
from torchvision import transforms
from NeuralNetworks.PyTorch.networks import TimeSeriesLearningSkip
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from stats import concordance_correlation_coefficient

def testNetwork(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wtgs = torch.load('TimeSeriesLearningSkip_best_loss.pth.tar')
    model.load_state_dict(wtgs['state_dict'])
    model.to(device)
    cccCalc = [[],[]]
    with torch.no_grad():
        model.eval()  # Set model to evaluate mode
        for inputs, labels, keypoints in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()
            outputs = model(inputs)
            labels = labels.reshape((labels.shape[0] * labels.shape[1], 1, labels.shape[2]))
            cccCalc[0].append(concordance_correlation_coefficient(outputs[:,:,0].cpu().numpy(),labels[:,:,0].cpu().numpy()))
            cccCalc[1].append(concordance_correlation_coefficient(outputs[:, :, 1].cpu().numpy(), labels[:, :, 1].cpu().numpy()))

        print(np.mean(cccCalc[0]))
        print(np.mean(cccCalc[1]))


def main():
    print("Initializing Datasets and Dataloaders...")
    input_size = (100,100)
    folders = ['Validation_Set']
    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
    ])
    model = TimeSeriesLearningSkip(sequenceSize=0)
    image_datasets = AFFData('/home/joaocardia/PycharmProjects/formated_aff', 'Validation_Set', data_transforms)
    dataloaders_dict = torch.utils.data.DataLoader(image_datasets, batch_size=25, shuffle=False, num_workers=0)
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    testNetwork(model, dataloaders_dict, criterion, optimizer_ft)

if __name__ == '__main__':
    main()
