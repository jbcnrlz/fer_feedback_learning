import argparse, os, face_alignment
from function import getFilePaths, saveLandmarks
from skimage import io

def main(args):
    files = getFilePaths(args.pathBase,imageExtesions=('.jpg'))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    for f in files:
        fileName = f.split(os.path.sep)[-1].split('.')[0]
        filePathDir = os.path.sep.join(f.split(os.path.sep)[:-1])
        if os.path.exists(os.path.join(filePathDir,fileName+'_landmarks.txt')):
            continue
        currIm = io.imread(f)
        print("Extracting landmarks from " + f)
        pred = fa.get_landmarks(currIm)
        if pred is None:
            continue
        saveLandmarks(pred[0],os.path.join(filePathDir,fileName+'_landmarks.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Landmarks')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    args = parser.parse_args()
    main(args)