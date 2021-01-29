import torch.utils.data as data, os, re, torch, numpy as np
from function import getDirectoriesInPath, getFilesInPath
from PIL import Image as im
#from generateDenseOpticalFlow import runDenseOpFlow

class AFFData(data.Dataset):
    def __init__(self, affData, phase, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []
        self.keypointsPath = []
        files = getFilesInPath(os.path.join(affData,phase))
        for r in files:
            fileName = r.split(os.path.sep)[-1]
            roi = int('right' in fileName)
            dirName = fileName.split('.')[0]
            if '_' in dirName:
                dirName = dirName.split('_')[0]

            valarrousal = self.loadLabels(r)
            for frameN, labelValue in enumerate(valarrousal):
                subjectData = os.path.join(affData,phase,dirName,'roi_%d_frame_%d.jpg') % (roi,frameN)
                if os.path.exists(subjectData):
                    self.filesPath.append(subjectData)
                    self.keypointsPath.append(os.path.join(affData, phase, dirName, 'roi_%d_frame_%d.txt') % (roi, frameN))
                    self.label.append(labelValue)

    def loadLabels(self,path):
        van = []
        with open(path,'r') as fp:
            fp.readline()
            for f in fp:
               van.append(list(map(float,f.split(','))))

        return van

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        path = self.filesPath[idx]
        image = im.open(path)
        label = self.label[idx]
        keypoints = self.keypointsPath[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(label)), keypoints


def getCASME2BlockData(casmepath,phase,blocksize):
    labels = []
    filesPath = []
    raw_data_loaded = getDirectoriesInPath(os.path.join(casmepath, phase))
    labelName = raw_data_loaded
    for r in raw_data_loaded:
        files = getFilesInPath(os.path.join(casmepath, phase, r))
        blockFiles = {}
        for f in files:
            fileName = '_'.join(f.split(os.path.sep)[-1].split('_')[:-1])
            if not fileName in blockFiles.keys():
                blockFiles[fileName] = []

            blockFiles[fileName].append(f)

        # qntdeBlock = int(sum([len(blockFiles[k]) for k in blockFiles]) / len(blockFiles.keys()) / self.blocksize)
        for k in blockFiles:
            blockFiles[k].sort(key=lambda f: int(re.sub('\D', '', f)))
            for nBl in range(len(blockFiles[k]) - blocksize):
                if (nBl + blocksize) > len(blockFiles[k]):
                    break
                blockData = blockFiles[k][nBl:nBl + blocksize]
                filesPath.append(blockData)
                labels.append(labelName.index(r))

    return labels, filesPath

class CASME2SALIENCY(data.Dataset):
    def __init__(self, casmepath, phase, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []
        raw_data_loaded = ['appex','neutral','offset','onset']
        labelName = raw_data_loaded
        for r in raw_data_loaded:
            files = getFilesInPath(os.path.join(casmepath,phase,r),onlyFiles=True,imagesOnly=True,imageExtesions=('png'))
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

class CASME2(data.Dataset):
    def __init__(self, casmepath, phase, transform=None):
        self.transform = transform
        self.label = []
        self.filesPath = []
        raw_data_loaded = getDirectoriesInPath(os.path.join(casmepath,phase))
        labelName = ['appex','neutral','offset','onset']
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

class CASME2Block(data.Dataset):
    def __init__(self, casmepath, phase, blocksize, transform):
        self.blocksize = blocksize
        self.transform = transform
        self.label = []
        self.filesPath = []
        raw_data_loaded = getDirectoriesInPath(os.path.join(casmepath,phase))
        labelName = ['appex','neutral','offset','onset']
        for r in raw_data_loaded:
            files = getFilesInPath(os.path.join(casmepath,phase,r))
            blockFiles = {}
            for f in files:
                fileName = '_'.join(f.split(os.path.sep)[-1].split('_')[:-1])
                if not fileName in blockFiles.keys():
                    blockFiles[fileName] = []

                blockFiles[fileName].append(f)

            #qntdeBlock = int(sum([len(blockFiles[k]) for k in blockFiles]) / len(blockFiles.keys()) / self.blocksize)
            for k in blockFiles:
                blockFiles[k].sort(key=lambda f: int(re.sub('\D', '', f)))
                for nBl in range(len(blockFiles[k]) - self.blocksize):
                    if (nBl+self.blocksize) > len(blockFiles[k]):
                        break
                    blockData = blockFiles[k][nBl:nBl+self.blocksize]
                    self.filesPath.append(blockData)
                    self.label.append(labelName.index(r))

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        blockOutput = None
        block = self.filesPath[idx]
        for path in block:
            image = im.open(path)
            label = self.label[idx]
            image = self.transform(image)
            if blockOutput is None:
                blockOutput = image
            else:
                blockOutput = torch.cat((blockOutput,image),0)


        return blockOutput, label

class CASME2BlockTemporal(data.Dataset):
    def __init__(self, casmepath, phase, blocksize, transform):
        self.blocksize = blocksize
        self.transform = transform
        self.label = []
        self.filesPath = []
        raw_data_loaded = getDirectoriesInPath(os.path.join(casmepath,phase))
        labelName = ['appex','neutral','offset','onset']
        for r in raw_data_loaded:
            files = getFilesInPath(os.path.join(casmepath,phase,r))
            blockFiles = {}
            for f in files:
                fileName = '_'.join(f.split(os.path.sep)[-1].split('_')[:-1])
                if not fileName in blockFiles.keys():
                    blockFiles[fileName] = []

                blockFiles[fileName].append(f)

            #qntdeBlock = int(sum([len(blockFiles[k]) for k in blockFiles]) / len(blockFiles.keys()) / self.blocksize)
            for k in blockFiles:
                blockFiles[k].sort(key=lambda f: int(re.sub('\D', '', f)))
                for nBl in range(len(blockFiles[k]) - self.blocksize):
                    if (nBl+self.blocksize) > len(blockFiles[k]):
                        break
                    blockData = blockFiles[k][nBl:nBl+self.blocksize]
                    self.filesPath.append(blockData)
                    self.label.append(labelName.index(r))

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):
        blockOutput = None
        block = self.filesPath[idx]
        for path in block:
            image = im.open(path)
            label = self.label[idx]
            image = self.transform(image)
            if blockOutput is None:
                blockOutput = image
            else:
                blockOutput = torch.cat((blockOutput,image),0)


        return blockOutput, label


class IngDiscLearnDataSet(data.Dataset):
    def __init__(self, idl_path, transform=None):
        self.phase = 'test'
        self.transform = transform
        self.idl_path = idl_path

        self.file_paths = getFilesInPath(self.idl_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = im.open(path)
        label = path

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class IngDiscLearnDataSetBlock(data.Dataset):
    def __init__(self, idl_path,blocksize, transform=None):
        self.phase = 'test'
        self.blocksize = blocksize
        self.filesPath = []
        self.transform = transform
        self.idl_path = idl_path

        raw_data_loaded = getFilesInPath(self.idl_path)
        blockFiles = {}
        for f in raw_data_loaded:
            fileName = '_'.join(f.split(os.path.sep)[-1].split('_')[:-1])
            if not fileName in blockFiles.keys():
                blockFiles[fileName] = []

            blockFiles[fileName].append(f)

        for k in blockFiles:
            blockFiles[k].sort(key=lambda f: int(re.sub('\D', '', f)))
            if len(blockFiles[k]) < blocksize:
                continue

            for nBl in range(len(blockFiles[k]) - 5):
                if (nBl+self.blocksize) > len(blockFiles[k]):
                    break
                blockData = blockFiles[k][nBl:nBl+self.blocksize]
                self.filesPath.append(blockData)

    def __len__(self):
        return len(self.filesPath)

    def __getitem__(self, idx):

        blockOutput = None
        label = []
        block = self.filesPath[idx]
        for path in block:
            image = im.open(path)
            label.append(path)
            image = self.transform(image)
            if blockOutput is None:
                blockOutput = image
            else:
                blockOutput = torch.cat((blockOutput,image),0)


        return blockOutput, label