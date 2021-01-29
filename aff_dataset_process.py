import cv2, shutil, os
import numpy as np
from function import getFilesInPath, getDirectoriesInPath
from PIL import Image
from facenet_pytorch import MTCNN

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


def main():
    batchSize = 50
    #if os.path.exists('formated_aff'):
    #    shutil.rmtree('formated_aff')

    detector = MTCNN(keep_all=True, device='cuda:0')
    #os.makedirs('formated_aff')
    #os.makedirs(os.path.join('formated_aff', 'Train_Set'))
    #os.makedirs(os.path.join('formated_aff', 'Validation_Set'))
    folders = getDirectoriesInPath("aff_dataset")
    for f in folders:
        videos = getFilesInPath(os.path.join('aff_dataset',f))
        for v in videos:
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
                while sucs:
                    sucs, imgv = vcap.read()
                    if sucs:
                        imgFaces = cv2.cvtColor(imgv, cv2.COLOR_BGR2RGB)
                        imgFaces = Image.fromarray(imgFaces)
                        frames.append(imgFaces)
                        frameIGV.append(imgv)
                        if len(frames) >= batchSize:
                            batch_boxes, _, batch_landmarks = detector.detect(frames, landmarks=True) #tim = imgv[a[1]:a[3],a[0]:a[2]]
                            for b_face, bbf in enumerate(batch_boxes):
                                if bbf is None:
                                    continue
                                faces = [(facc[0],facc[1],facc[2]-facc[0],facc[3]-facc[1]) for facc in bbf.astype(int)]
                                newROIs = isNewROI(rois, faces)
                                for (x, y, w, h) in newROIs:
                                    rois.append((x, y, w, h))
                                roins = findROI(rois, faces)
                                for fnum, b in enumerate(faces):
                                    if frameIGV[b_face] is None or b is None:
                                        continue
                                    if b[0] < 0 or b[1] < 0:
                                        continue
                                    fImage = frameIGV[b_face][b[1]:b[1] + b[3],b[0]:b[0] + b[2]]
                                    cv2.imwrite(os.path.join('formated_aff', f, fileName[:-4],
                                                             "roi_" + str(roins[fnum]) + "_frame_" + str(
                                                                 (int(frame_number / batchSize) * batchSize) + b_face) + ".jpg"), fImage)
                            frames = []
                            frameIGV = []
                    else:
                        if len(frames) > 0:
                            batch_boxes, _, batch_landmarks = detector.detect(frames,landmarks=True)  # tim = imgv[a[1]:a[3],a[0]:a[2]]
                            for b_face, bbf in enumerate(batch_boxes):
                                if bbf is None:
                                    continue
                                faces = [(facc[0], facc[1], facc[2] - facc[0], facc[3] - facc[1]) for facc in bbf.astype(int)]
                                newROIs = isNewROI(rois, faces)
                                for (x, y, w, h) in newROIs:
                                    rois.append((x, y, w, h))
                                roins = findROI(rois, faces)
                                for fnum, b in enumerate(faces):
                                    if frameIGV[b_face] is None or b is None:
                                        continue
                                    if b[0] < 0 or b[1] < 0:
                                        continue
                                    #fImage = np.array(frames[b_face].crop((b[0],b[1],b[0] + b[2],b[1] + b[3])))
                                    fImage = frameIGV[b_face][b[1]:b[1] + b[3], b[0]:b[0] + b[2]]
                                    cv2.imwrite(os.path.join('formated_aff', f, fileName[:-4],
                                                             "roi_" + str(roins[fnum]) + "_frame_" + str(
                                                                 (int(
                                                                     frame_number / batchSize) * batchSize) + b_face) + ".jpg"),
                                                fImage)
                    '''
                    #faces = detectFace(face_cascade,imgv)
                    faces = [f['box'] for f in detector.detect_faces(imgFaces)]
                    newROIs = isNewROI(rois,faces)
                    for (x,y,w,h) in newROIs:
                        rois.append((x,y,w,h))

                    roins = findROI(rois,faces)

                    for fnum, b in enumerate(faces):
                        print("Extraindo face %d" % (roins[fnum]))
                        fImage = imgv[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
                        cv2.imwrite(os.path.join('formated_aff',f,fileName[:-4],"roi_" + str(roins[fnum]) + "_frame_"+str(frame_number)+".jpg"),fImage)
                    '''
                    frame_number+=1



if __name__ == '__main__':
    main()