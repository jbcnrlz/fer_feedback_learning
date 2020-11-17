import torch, time, copy
from torchvision import transforms, models
from DatasetClasses.FEDatasets import CASME2
from torch import nn, optim

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def main():# Data augmentation and normalization for training

    print("Initializing Datasets and Dataloaders...")
    resnetFT, input_size = initialize_model(4,True)
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
    # Create training and validation datasets
    image_datasets = {x: CASME2('CASME2_formated', x,data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    print("Params to learn:")
    params_to_update = []
    for name, param in resnetFT.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    a,b = trainResnet(resnetFT,dataloaders_dict,criterion,optimizer_ft)


'''
def main():
    if os.path.exists('frames_face'):
        shutil.rmtree('frames_face')

    os.makedirs('frames_face')
    det = MTCNN()
    sucs = 1
    vcap = cv2.VideoCapture("2GRAV1qua1.mp4")
    frame_number=0
    while sucs:
        sucs, imgv = vcap.read()
        faces = det.detect_faces(imgv)
        for fnum, b in enumerate(faces):
            print("Extraindo face %d" % (fnum))
            #cv2.rectangle(imgv,(b['box'][0],b['box'][1]),(b['box'][0]+b['box'][2],b['box'][1]+b['box'][3]),(255,0,0))
            fImage = imgv[b['box'][1]:b['box'][1]+b['box'][3],b['box'][0]:b['box'][0]+b['box'][2]]
            cv2.imwrite(os.path.join('frames_face',"face_" + str(fnum) + "_frame_"+str(frame_number)+".jpg"),fImage)
        frame_number+=1
'''
def trainResnet(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                fName = '%s_best_rank.pth.tar' % ('resnet_50')
                torch.save({
                    'epoch': epoch,
                    'arch': 'resnet',
                    'state_dict': best_model_wts
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
'''
def trainResnet():
    resNet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
'''
if __name__ == '__main__':
    main()
