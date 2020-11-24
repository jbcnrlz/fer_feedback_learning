from NeuralNetworks.PyTorch.networks import TimeSeriesLearning
from torchvision import transforms
from function import getDirectoriesInPath
import torch,os,shutil
from DatasetClasses.FEDatasets import IngDiscLearnDataSetBlock

def createFoldersForEmotions(emotions,pathBase='expressionsEmotions'):
    if os.path.exists(pathBase):
        shutil.rmtree(pathBase)

    for e in emotions:
        os.makedirs(os.path.join(pathBase,e))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wtgs = torch.load('temporal_resnet_50_best_loss.pth.tar')
    rnet = TimeSeriesLearning(4,15)
    rnet = rnet.to(device)
    valT = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valDataset = torch.utils.data.DataLoader(IngDiscLearnDataSetBlock('frames_face',transform = valT,blocksize=5), batch_size=20, shuffle=False, num_workers=4)
    rnet.load_state_dict(wtgs['state_dict'])
    classes = getDirectoriesInPath(os.path.join('CASME2_formated','train'))
    createFoldersForEmotions(classes)
    blockNumber = 0
    with torch.no_grad():
        rnet.eval()
        for batch_i, (imgs, targets) in enumerate(valDataset):
            outputs = rnet(imgs.cuda())
            _, predicts = torch.max(outputs, 1)
            predictsCPU = predicts.cpu()
            for idx, t in enumerate(targets):
                for ft in t:
                    fileName = ft.split(os.path.sep)[-1]
                    shutil.copyfile(ft,os.path.join('expressionsEmotions',classes[predictsCPU[idx].item()],'block_'+str(blockNumber)+'_'+fileName))
                blockNumber += 1


if __name__ == '__main__':
    main()