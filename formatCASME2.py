import pandas as pd, os, shutil, random, cv2
#from mtcnn import MTCNN

def detectAndCropFace(cascade,faceImage):
    grayIm = cv2.cvtColor(faceImage,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(grayIm, 1.1, 4)
    for (x, y, w, h)  in faces:
        return faceImage[y:y+h, x:x+w]

def main():
    face_cascade = cv2.CascadeClassifier('cascadeFolder/haarcascade_frontalface_default.xml')
    csvLoaded = pd.read_csv('CASME2-RAW/CASME2_coding-20190701.csv')
    if os.path.exists('CASME2_formated'):
        shutil.rmtree('CASME2_formated')

    for phase in ['train','val']:
        os.makedirs(os.path.join('CASME2_formated',phase,'neutral'))
        os.makedirs(os.path.join('CASME2_formated',phase,'onset'))
        os.makedirs(os.path.join('CASME2_formated',phase,'set'))
        os.makedirs(os.path.join('CASME2_formated',phase,'offset'))

    valData = int(csvLoaded.shape[0] * 0.1)

    for i in range(csvLoaded.shape[0]):

        phase = 'train'
        if (random.randint(0,1) == 1) and valData > 0:
            valData -= 1
            phase = 'val'

        folderPath = os.path.join('CASME2-RAW','CASME2-RAW','sub%02d' % (csvLoaded['Subject'][i]),csvLoaded['Filename'][i])

        #shutil.copyfile(os.path.join(folderPath, 'img%d.jpg' % (csvLoaded['OnsetFrame'][i])),
        #                os.path.join('CASME2_formated',phase, 'onset','%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],csvLoaded['OnsetFrame'][i])))

        imageOp = cv2.imread(os.path.join(folderPath, 'img%d.jpg' % (csvLoaded['OnsetFrame'][i])))
        fImage = detectAndCropFace(face_cascade,imageOp)
        cv2.imwrite(os.path.join('CASME2_formated',phase, 'onset','%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],csvLoaded['OnsetFrame'][i])), fImage)

        #shutil.copyfile(os.path.join(folderPath, 'img%d.jpg' % (csvLoaded['OffsetFrame'][i])),
        #                os.path.join('CASME2_formated',phase, 'offset', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], csvLoaded['OffsetFrame'][i])))

        imageOp = cv2.imread(os.path.join(folderPath, 'img%d.jpg' % (csvLoaded['OffsetFrame'][i])))
        fImage = detectAndCropFace(face_cascade,imageOp)
        cv2.imwrite(os.path.join('CASME2_formated',phase, 'offset','%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],csvLoaded['OnsetFrame'][i])), fImage)

        for j in range(csvLoaded['OnsetFrame'][i]+1,csvLoaded['OffsetFrame'][i]):
            #shutil.copyfile(os.path.join(folderPath,'img%d.jpg' % j),
            #                os.path.join('CASME2_formated',phase, 'set', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],j)))
            imageOp = cv2.imread(os.path.join(folderPath, 'img%d.jpg' % (j)))
            fImage = detectAndCropFace(face_cascade,imageOp)
            cv2.imwrite(os.path.join('CASME2_formated',phase, 'set', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],j)), fImage)

        for j in range(1,csvLoaded['OnsetFrame'][i]):
            #shutil.copyfile(os.path.join(folderPath,'img%d.jpg' % j),
            #                os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j)))
            imageOp = cv2.imread(os.path.join(folderPath, 'img%d.jpg' % (j)))
            fImage = detectAndCropFace(face_cascade,imageOp)
            cv2.imwrite(os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],j)), fImage)

        j = csvLoaded['OffsetFrame'][i]+1
        while True:
            if not os.path.exists(os.path.join(folderPath,'img%d.jpg' % j)):
                break
            #shutil.copyfile(
            #    os.path.join(folderPath,'img%d.jpg' % j),
            #    os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i], j))
            #)
            imageOp = cv2.imread(os.path.join(folderPath, 'img%d.jpg' % (j)))
            fImage = detectAndCropFace(face_cascade,imageOp)
            cv2.imwrite(os.path.join('CASME2_formated',phase, 'neutral', '%02d_%s_%d.jpg' % (csvLoaded['Subject'][i],csvLoaded['Filename'][i],j)), fImage)
            j += 1

if __name__ == '__main__':
    main()