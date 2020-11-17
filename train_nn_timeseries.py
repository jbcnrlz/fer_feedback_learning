from finetune_resnet import set_parameter_requires_grad
from torchvision import transforms, models
from DatasetClasses.FEDatasets import CASME2Block
from torch import nn, optim
import torch, copy, time
from NeuralNetworks.PyTorch.networks import TimeSeriesLearning

def temporalRESNET(inputChannels=15):
    model_ft = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    model_ft.conv1 = nn.Conv2d(inputChannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.Dropout(),
        nn.Linear(1024,4)
    )
    input_size = 224

    return model_ft, input_size

def trainNetwork(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                fName = '%s_best_rank.pth.tar' % ('temporal_resnet_50')
                torch.save({
                    'epoch': epoch,
                    'arch': 'resnet',
                    'state_dict': best_model_wts
                }, fName)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_loss = copy.deepcopy(model.state_dict())
                fName = '%s_best_loss.pth.tar' % ('temporal_resnet_50')
                torch.save({
                    'epoch': epoch,
                    'arch': 'resnet',
                    'state_dict': best_model_loss
                }, fName)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main():# Data augmentation and normalization for training

    print("Initializing Datasets and Dataloaders...")
    #resnetFT, input_size = temporalRESNET()
    resnetFT = TimeSeriesLearning(4,64,224*224*15)
    input_size = (224,224)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: CASME2Block('CASME2_formated', x, 5, data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=25, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
    print("Params to learn:")
    params_to_update = []
    for name, param in resnetFT.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    a,b = trainNetwork(resnetFT,dataloaders_dict,criterion,optimizer_ft)

if __name__ == '__main__':
    main()