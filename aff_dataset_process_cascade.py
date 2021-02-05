import cv2, shutil, os, numpy as np, face_alignment
from function import getFilesInPath, getDirectoriesInPath, saveLandmarks
import multiprocessing

def findROI(rois,areas):
    returnData = []
    for (fx, fy, fw, fh) in areas:
        for rn, (cx, cy, cw, ch) in enumerate(rois):
            dist = np.linalg.norm(np.array([cx,cy]) - np.array([fx,fy]))
            if (dist < (cx+cw)) and (dist < (cy+ch)):
                returnData.append(rn)
                break

    return returnData

def detectFace(cascade,faceImage):
    try:
        grayIm = cv2.cvtColor(faceImage,cv2.COLOR_BGR2GRAY)
        return cascade.detectMultiScale(grayIm, 1.1, 4)
    except:
        return []

def isNewROI(currRoi,foundFaces):
    returnData = []
    for (fx, fy, fw, fh) in foundFaces:
        colision = 0
        for (cx, cy, cw, ch) in currRoi:
            dist = np.linalg.norm(np.array([cx,cy]) - np.array([fx,fy]))
            if (dist < (cx+cw)) and (dist < (cy+ch)):
                colision = 1

        if not colision:
            returnData.append((fx, fy, fw, fh))

    return returnData

def preProcessWithThread(v,f,fa,face_cascade):
    fileName = v.split(os.path.sep)[-1]
    if v.endswith('txt'):
        shutil.copyfile(os.path.join('aff_dataset', f, fileName[:-3] + 'txt'),
                        os.path.join('formated_aff', f, fileName[:-3] + 'txt'))
    if v.endswith('mp4'):
        if not os.path.exists(os.path.join('formated_aff', f, fileName[:-4])):
            os.makedirs(os.path.join('formated_aff', f, fileName[:-4]))

        vcap = cv2.VideoCapture(v)
        frame_number = 0
        rois = []
        sucs = True
        print('Video - %s' % (v))
        while sucs:
            sucs, imgv = vcap.read()
            for roicheck in range(10):
                fileNameROI = os.path.join('formated_aff', f, fileName[:-4],
                             "roi_" + str(roicheck) + "_frame_" + str(
                                 frame_number) + '.jpg')
                if os.path.exists(fileNameROI):
                    frame_number += 1
                    break
            else:

                if sucs:
                    try:
                        faces = detectFace(face_cascade, imgv)
                        newROIs = isNewROI(rois, faces)
                        for (x, y, w, h) in newROIs:
                            rois.append((x, y, w, h))

                        roins = findROI(rois, faces)

                        for fnum, b in enumerate(faces):
                            fImage = imgv[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]
                            if min(fImage.shape[:-1]) < 100:
                                continue
                            isReallyFace = fa.get_landmarks(fImage)
                            if not isReallyFace is None:
                                saveLandmarks(isReallyFace[0], os.path.join('formated_aff', f, fileName[:-4],
                                                                            "roi_" + str(roins[fnum]) + "_frame_" + str(
                                                                                frame_number) + '.txt'))
                                cv2.imwrite(os.path.join('formated_aff', f, fileName[:-4],
                                                         "roi_" + str(roins[fnum]) + "_frame_" + str(
                                                             frame_number) + '.jpg'), fImage)
                    except:
                        print("Erro")

                frame_number += 1


def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    #if os.path.exists('formated_aff'):
    #    shutil.rmtree('formated_aff')

    face_cascade = cv2.CascadeClassifier('cascadeFolder/haarcascade_frontalface_default.xml')
    #os.makedirs('formated_aff')
    #os.makedirs(os.path.join('formated_aff', 'Train_Set'))
    #os.makedirs(os.path.join('formated_aff', 'Validation_Set'))
    folders = getDirectoriesInPath("aff_dataset")
    for f in folders:
        videos = getFilesInPath(os.path.join('aff_dataset',f))
        for v in videos:
            preProcessWithThread(v,f,fa,face_cascade)
            '''
            fileName = v.split(os.path.sep)[-1]
            if v.endswith('txt'):
                shutil.copyfile(os.path.join('aff_dataset', f, fileName[:-3] + 'txt'),
                                os.path.join('formated_aff', f, fileName[:-3] + 'txt'))
            if v.endswith('mp4'):
                if not os.path.exists(os.path.join('formated_aff',f,fileName[:-4])):
                    os.makedirs(os.path.join('formated_aff',f,fileName[:-4]))

                vcap = cv2.VideoCapture(v)
                frame_number=0
                rois = []
                frames = []
                frameIGV = []
                sucs = True
                print('Video - %s' % (v))
                while sucs:
                    sucs, imgv = vcap.read()
                    if sucs:
                        try:
                            faces = detectFace(face_cascade, imgv)
                            newROIs = isNewROI(rois, faces)
                            for (x, y, w, h) in newROIs:
                                rois.append((x, y, w, h))

                            roins = findROI(rois, faces)

                            for fnum, b in enumerate(faces):
                                fImage = imgv[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]
                                if min(fImage.shape[:-1]) < 100:
                                    continue
                                isReallyFace = fa.get_landmarks(fImage)
                                if not isReallyFace is None:
                                    saveLandmarks(isReallyFace[0],os.path.join('formated_aff', f, fileName[:-4],"roi_" + str(roins[fnum]) + "_frame_" + str(frame_number) + '.txt'))
                                    cv2.imwrite(os.path.join('formated_aff', f, fileName[:-4],"roi_" + str(roins[fnum]) + "_frame_" + str(frame_number) + '.jpg'), fImage)
                        except:
                            print("Erro")

                    frame_number+=1
            '''


if __name__ == '__main__':
    main()