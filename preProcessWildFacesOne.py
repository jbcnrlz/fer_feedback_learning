import cv2, shutil, os, argparse
import numpy as np
from function import getFilesInPath, getDirectoriesInPath
from scipy.spatial.distance import euclidean
from PIL import Image
from facenet_pytorch import MTCNN

def readFile(path):
    annot = []
    with open(path,'r') as fan:
        linesProc = fan.readlines()[3:-1]
        for l in linesProc:
            annot.append(list(map(float,l.strip().split(" "))))

    return annot

def openLandmarks(pathInicial,dir,videoName):
    pathFile = os.path.join(pathInicial,'landmarks',dir,videoName[:-4])
    dirs = getFilesInPath(pathFile)
    landMarks = {}
    for d in dirs:
        idxFrameLand = int(d.split(os.path.sep)[-1][:-4])
        landMarks[idxFrameLand] = readFile(d)

    return landMarks

def getBoundingbox(landmarks):
    lnds = np.array(landmarks)
    x = int(lnds[:,0].min())
    y = int(lnds[:,1].min())
    x2 = int(lnds[:,0].max())
    y2 = int(lnds[:,1].max())
    boost = int(euclidean(lnds[33],lnds[8]))
    return (x,y-boost,x2,y2)

def main():

    parser = argparse.ArgumentParser(description='Organize AffWild1')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--whereTo', help='Where to save', required=True)
    parser.add_argument('--deleteOld', help='Remove old dataset?', default=1, type=int)
    args = parser.parse_args()

    if args.deleteOld:
        if os.path.exists(args.whereTo):
            shutil.rmtree(args.whereTo)

        os.makedirs(args.whereTo)

    dirsFaces = getDirectoriesInPath(os.path.join(args.pathBase,'videos'))
    for d in dirsFaces:
        print("Extracting from %s" % (d))
        if not os.path.exists(os.path.join(args.whereTo,d)):
            os.makedirs(os.path.join(args.whereTo,d))

        filesFace = getFilesInPath(os.path.join(args.pathBase,'videos',d))
        for f in filesFace:
            fileName = f.split(os.path.sep)[-1]
            print("Doing video %s" % (fileName))

            if not os.path.exists(os.path.join(args.whereTo, d,fileName[:-4])):
                os.makedirs(os.path.join(args.whereTo, d,fileName[:-4]))

            faceAnn = openLandmarks(args.pathBase, d,fileName)
            vcap = cv2.VideoCapture(f)

            frame_number = 0
            sucs = True
            while sucs:
                sucs, imgv = vcap.read()
                if sucs and frame_number in faceAnn.keys():
                    if not os.path.exists(os.path.join(args.whereTo, d,fileName[:-4],str(frame_number) + '.jpg')):
                        x1,y1,x2,y2 = getBoundingbox(faceAnn[frame_number])
                        fImage = imgv[y1:y2, x1:x2]
                        if len(fImage) > 0:
                            try:
                                cv2.imwrite(os.path.join(args.whereTo, d,fileName[:-4],str(frame_number) + '.jpg'),fImage)
                            except:
                                print('Error in save frame %d'%(frame_number))
                frame_number += 1


if __name__ == '__main__':
    main()