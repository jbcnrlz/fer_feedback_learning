import argparse, os, face_alignment
from function import getFilePaths
from skimage import io

def saveLandmarks(ldns,filepath):
    with open(filepath,'w') as fp:
        for l in ldns:
            fp.write(','.join(list(map(str,l)))+'\n')

def main(args):
    files = getFilePaths(args.pathBase,imageExtesions=('.jpg'))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    for f in files:
        print("Extracting landmarks from " + f)
        currIm = io.imread(f)
        fileName = f.split(os.path.sep)[-1].split('.')[0]
        filePathDir = os.path.sep.join(f.split(os.path.sep)[:-1])
        if os.path.exists(os.path.join(filePathDir,fileName+'_landmarks.txt')):
            continue
        pred = fa.get_landmarks(currIm)
        if pred is None:
            continue
        saveLandmarks(pred[0],os.path.join(filePathDir,fileName+'_landmarks.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Landmarks')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    args = parser.parse_args()
    main(args)