import os

def getDirectoriesInPath(path):
    return [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]


def getFilesInPath(path, onlyFiles=True, fullPath=True):
    if fullPath:
        joinFunc = lambda p, fn: os.path.join(p, fn)
    else:
        joinFunc = lambda p, fn: fn

    if onlyFiles:
        return [joinFunc(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [joinFunc(path, f) for f in os.listdir(path)]