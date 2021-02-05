from NeuralNetworks.PyTorch.networks import TimeSeriesLearningSkip
from torchvision import transforms
from function import getDirectoriesInPath
import torch,os,shutil
from DatasetClasses.FEDatasets import IngDiscLearnDataSet

def createFoldersForEmotions(emotions,pathBase='expressionsEmotions'):
    if os.path.exists(pathBase):
        shutil.rmtree(pathBase)

    for e in emotions:
        os.makedirs(os.path.join(pathBase,e))

def outputLabel(pathFile,fileName,valArr):
    with open(pathFile,'a') as labelFile:
        for idx, f in enumerate(fileName):
            labelFile.write(str(valArr[idx][0][0].tolist())+ ',' + str(valArr[idx][0][1].tolist()) + ','+f+'\n')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wtgs = torch.load('TimeSeriesLearningSkip_best_loss.pth.tar')
    rnet = TimeSeriesLearningSkip()
    rnet = rnet.to(device)
    valT = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    valDataset = torch.utils.data.DataLoader(IngDiscLearnDataSet('clipe1',transform = valT), batch_size=20, shuffle=False, num_workers=4)
    rnet.load_state_dict(wtgs['state_dict'])
    with torch.no_grad():
        rnet.eval()
        for batch_i, (imgs, targets) in enumerate(valDataset):
            outputs = rnet(imgs.cuda())
            outputLabel('resultado_clipe1.txt',targets,outputs.cpu())



if __name__ == '__main__':
    main()