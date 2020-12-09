from function import getFilePaths, arrangeFaces, readFeturesFiles
import argparse, cv2, numpy as np, os, re, shutil

def paintLandmark(image,landmarks,color=(0,0,255)):
    for landmark in landmarks:
        for x, y in landmark:
            # display landmarks on "image_cropped"
            # with white colour in BGR and thickness 1
            cv2.circle(image, (x, y), 1, color, 1)

    return image

def generateFeatureLandmarks(zeroMatrix,files):
    for f in files:
        fpath = f
        if fpath[:-3] != 'txt':
            fpath = fpath[:-4] + '_landmarks.txt'
        if os.path.exists(fpath):
            landMarks = np.array(readFeturesFiles(fpath),dtype=np.uint8)
            for l in landMarks:
                if l[0] < zeroMatrix.shape[0] and l[1] < zeroMatrix.shape[1]:
                    zeroMatrix[l[0]][l[1]] += 1

    return cv2.normalize(zeroMatrix.T, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

def main():
    parser = argparse.ArgumentParser(description='Saliency')
    parser.add_argument('--pathBase', help='Path for faces', required=True)
    parser.add_argument('--blocksize', help='Size of the block', default=5, type=int)
    parser.add_argument('--folderImages', help='Folder for FE maps', default='CASME_landmarks')
    args = parser.parse_args()

    if os.path.exists(args.folderImages):
        shutil.rmtree(args.folderImages)

    os.makedirs(args.folderImages)

    files = getFilePaths(args.pathBase,imageExtesions=('.jpg'))
    arrFiles = arrangeFaces(files)
    for a in arrFiles:
        arrFiles[a].sort(key=lambda f: int(re.sub('\D', '', f)))
        if len(arrFiles[a]) > (args.blocksize * 2)+1:
            qntity = len(arrFiles) - (args.blocksize * 2)
            for f in range(args.blocksize + 1,args.blocksize + qntity,1):
                if f >= len(arrFiles[a]):
                    break
                print("Generating " + arrFiles[a][f])
                folder = arrFiles[a][f].split(os.path.sep)[:-1]
                fileName = arrFiles[a][f].split(os.path.sep)[-1]
                folder = args.folderImages + os.path.sep + os.path.sep.join(folder[1:])
                if not os.path.exists(folder):
                    os.makedirs(folder)

                imageMd = cv2.imread(arrFiles[a][f])
                imageMd = cv2.cvtColor(imageMd,cv2.COLOR_BGR2GRAY)
                beforeIm = generateFeatureLandmarks(np.zeros_like(imageMd),arrFiles[a][:f])
                afterIm = generateFeatureLandmarks(np.zeros_like(imageMd), arrFiles[a][f+1:])
                #cv2.imwrite(os.path.join(folder,fileName[:-4]+'_before_lands.jpg'),beforeIm)
                #cv2.imwrite(os.path.join(folder, fileName[:-4] + '_after_lands.jpg'), beforeIm)
                if (np.sum(beforeIm) > 0) and (np.sum(afterIm) > 0):
                    cv2.imwrite(os.path.join(folder,fileName), np.dstack((beforeIm, imageMd, afterIm)))


if __name__ == '__main__':
    main()