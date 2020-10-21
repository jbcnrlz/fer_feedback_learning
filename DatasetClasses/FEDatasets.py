import torch.utils.data as data, os, cv2
from function import getDirectoriesInPath, getFilesInPath
from PIL import Image as im

class CASME2(data.Dataset):
    def __init__(self, casmepath, phase, transform=None):
        self.transform = transform
        labelName = ['neutral','offset','onset','set']
        self.label = []
        self.filesPath = []
        raw_data_loaded = getDirectoriesInPath(os.path.join(casmepath,phase))
        for r in raw_data_loaded:
            files = getFilesInPath(os.path.join(casmepath,phase,r))
            for f in files:
                self.filesPath.append(f)
                self.label.append(labelName.index(r))

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = self.label[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label