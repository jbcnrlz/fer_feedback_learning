from finetune_resnet import initialize_model
from torchvision import transforms
import torch,os,shutil
from DatasetClasses.FEDatasets import IngDiscLearnDataSet

def createFoldersForEmotions(emotions,pathBase='expressionsEmotions'):
    if os.path.exists(pathBase):
        shutil.rmtree(pathBase)

    for e in emotions:
        os.makedirs(os.path.join(pathBase,e))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wtgs = torch.load('temporal_resnet_50_best_rank.pth.tar')
    rnet, imsize = initialize_model(4,True,False)
    rnet = rnet.to(device)
    valT = transforms.Compose([
        transforms.Resize((imsize,imsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valDataset = torch.utils.data.DataLoader(IngDiscLearnDataSet('frames_face',transform = valT), batch_size=20, shuffle=True, num_workers=4)
    rnet.load_state_dict(wtgs['state_dict'])
    classes = ['neutral', 'offset', 'onset', 'set']
    createFoldersForEmotions(classes)
    with torch.no_grad():
        rnet.eval()
        for batch_i, (imgs, targets) in enumerate(valDataset):
            print('opra')
            outputs = rnet(imgs.cuda())
            _, predicts = torch.max(outputs, 1)
            predictsCPU = predicts.cpu()
            for idx, t in enumerate(targets):
                fileName = t.split(os.path.sep)[-1]
                shutil.copyfile(t,os.path.join('expressionsEmotions',classes[predictsCPU[idx].item()],fileName))


if __name__ == '__main__':
    main()