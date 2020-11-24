import pandas as pd, os, shutil, random, cv2
#from mtcnn import MTCNN
from extractFacesFromVideo import *

def detectAndCropFace(cascade,faceImage):
    grayIm = cv2.cvtColor(faceImage,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grayIm, 1.1, 4)
    for (x, y, w, h)  in faces:
        return faceImage[y:y+h, x:x+w]

def cutFace(image,face_cascade,pathSave):
    imOpened = cv2.imread(image)
    facePos = detectFace(face_cascade, imOpened)
    for (x, y, w, h) in facePos:
        fImage = imOpened[y:y + h, x:x + w]
        resized = cv2.resize(fImage,(224,224),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(pathSave, resized)

def main():
    face_cascade = cv2.CascadeClassifier('cascadeFolder/haarcascade_frontalface_default.xml')
    csvLoaded = pd.read_csv('CASME2-RAW/CASME2_coding-20190701.csv')
    if os.path.exists('CASME2_formated'):
        shutil.rmtree('CASME2_formated')

    for phase in ['train','val']:
        os.makedirs(os.path.join('CASME2_formated',phase,'neutral'))
        os.makedirs(os.path.join('CASME2_formated',phase,'onset'))
        os.makedirs(os.path.join('CASME2_formated',phase,'appex'))
        os.makedirs(os.path.join('CASME2_formated',phase,'offset'))

    valData = int(csvLoaded.shape[0] * 0.1)

    for i in range(csvLoaded.shape[0]):

        phase = 'train'
        if (random.randint(0,1) == 1) and valData > 0:
            valData -= 1
            phase = 'val'

        sizePhases = int((csvLoaded['OffsetFrame'][i] - csvLoaded['OnsetFrame'][i]) / 3.0)
        imageNumberCopy = [
            ('onset', csvLoaded['OnsetFrame'][i]),
            ('appex' ,csvLoaded['OnsetFrame'][i] + sizePhases),
            ('offset', csvLoaded['OffsetFrame'][i] - sizePhases),
            ('stop' , csvLoaded['OffsetFrame'][i])
        ]
        folderPath = os.path.join('CASME2-RAW','CASME2-RAW','sub%02d' % (csvLoaded['Subject'][i]),csvLoaded['Filename'][i])

        for idx, it in enumerate(imageNumberCopy[:-1]):
            for fNum in range(it[1],imageNumberCopy[idx+1][1]):
                cutFace(os.path.join(folderPath, 'img%d.jpg' % (fNum)),face_cascade,os.path.join('CASME2_formated', phase, it[0],
                                 '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i], csvLoaded['Filename'][i], fNum)))
                #shutil.copyfile(
                #    os.path.join(folderPath, 'img%d.jpg' % (fNum)),
                #    os.path.join('CASME2_formated',phase, it[0],'%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],fNum))
                #)

        for j in range(1,csvLoaded['OnsetFrame'][i]):
            cutFace(os.path.join(folderPath,'img%d.jpg' % j), face_cascade,
                    os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j)))
            #shutil.copyfile(
            #    os.path.join(folderPath,'img%d.jpg' % j),
            #    os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j))
            #)

        j = csvLoaded['OffsetFrame'][i]
        while True:
            if not os.path.exists(os.path.join(folderPath,'img%d.jpg' % j)):
                break
            cutFace(os.path.join(folderPath,'img%d.jpg' % j), face_cascade,
                    os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j)))
            #shutil.copyfile(
            #    os.path.join(folderPath,'img%d.jpg' % j),
            #    os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j))
            #)
            j += 1

if __name__ == '__main__':
    main()