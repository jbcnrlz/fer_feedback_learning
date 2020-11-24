import os

def getDirectoriesInPath(path):
    return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]


def getFilesInPath(path, onlyFiles=True, fullPath=True,imagesOnly=False):
    if fullPath:
        joinFunc = lambda p, fn: os.path.join(p, fn)
    else:
        joinFunc = lambda p, fn: fn

    if onlyFiles:
        return [joinFunc(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and (not imagesOnly or f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))))]
    else:
        return [joinFunc(path, f) for f in os.listdir(path) if (not imagesOnly or f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')))]