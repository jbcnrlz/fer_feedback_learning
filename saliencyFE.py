from function import getFilePaths, arrangeFaces
import argparse, cv2, numpy as np, os, re

def main():
    parser = argparse.ArgumentParser(description='Saliency')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    args = parser.parse_args()

    sal = cv2.saliency.StaticSaliencySpectralResidual_create()

    files = getFilePaths(args.pathBase)
    arrFiles = arrangeFaces(files)
    for a in arrFiles:
        arrFiles[a].sort(key=lambda f: int(re.sub('\D', '', f)))
        prevSalMap = None
        currSalMap = None
        for f in arrFiles[a]:
            im = cv2.imread(f)
            (success, saliencyMap) = sal.computeSaliency(im)
            saliencyMap = cv2.normalize(saliencyMap * 255, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            if prevSalMap is None:
                prevSalMap = saliencyMap
                continue

            if currSalMap is None:
                currSalMap = saliencyMap
                continue

            gsim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

            fileName = f.split(os.path.sep)[-1].split('.')[0] + '_sal_map.png'
            filePat = os.path.sep.join(f.split(os.path.sep)[:-1])
            print("Generating filename: "+fileName)
            cv2.imwrite(os.path.join(filePat,fileName),np.dstack((prevSalMap,currSalMap,saliencyMap)))
            prevSalMap = currSalMap
            currSalMap = saliencyMap


if __name__ == '__main__':
    main()