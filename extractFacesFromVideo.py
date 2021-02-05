import cv2, shutil, os
import numpy as np

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
    grayIm = cv2.cvtColor(faceImage,cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(grayIm, 1.1, 4)

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
    if os.path.exists('clipe1'):
        shutil.rmtree('clipe1')

    os.makedirs('clipe1')
    sucs = 1
    vcap = cv2.VideoCapture("clipe1.mp4")
    frame_number=0
    face_cascade = cv2.CascadeClassifier('cascadeFolder/haarcascade_frontalface_default.xml')
    rois = []
    while sucs:
        sucs, imgv = vcap.read()
        faces = detectFace(face_cascade,imgv)
        newROIs = isNewROI(rois,faces)
        for (x,y,w,h) in newROIs:
            rois.append((x,y,w,h))

        roins = findROI(rois,faces)

        for fnum, b in enumerate(faces):
            print("Extraindo face %d" % (roins[fnum]))
            fImage = imgv[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
            cv2.imwrite(os.path.join('clipe1',"roi_" + str(roins[fnum]) + "_frame_"+str(frame_number)+".jpg"),fImage)
        frame_number+=1



if __name__ == '__main__':
    main()