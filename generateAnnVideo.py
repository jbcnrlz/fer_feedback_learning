from NeuralNetworks.PyTorch.networks import TimeSeriesLearningSkip
import cv2, shutil, os, torch
import numpy as np, random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from DatasetClasses.FEDatasets import IngDiscLearnDataSetBlock
from matplotlib.figure import Figure

def findROI(rois,areas):
    returnData = []
    for (fx, fy, fw, fh) in areas:
        for rn, (cx, cy, cw, ch) in enumerate(rois):
            #dist = np.linalg.norm(np.array([cx,cy]) - np.array([fx,fy]))
            if (fx < (cx+fw)) and ((fx + cw) > cx) and (fy < (cy + fh)) and ((fh + fy) > cy):
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

def getRandomColors():
    #return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    return (255,0,0)

def getArousalValence(faceImgs,model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    faceImgsTensor = torch.FloatTensor(faceImgs)
    faceImgsTensor = faceImgsTensor.transpose(2,3).transpose(1,2)
    faceImgsTensor = faceImgsTensor.reshape((1,120,100,100))

    faceImgsTensor = faceImgsTensor.to(device)

    results = model(faceImgsTensor)
    results = results.cpu()
    a = results[:,:,0].flatten()
    v = results[:,:,1].flatten()
    del faceImgsTensor, results
    torch.cuda.empty_cache()

    return a, v


def getResult(valence,arousal):
    if valence < 0 and arousal > 0:
        return 'intense_unpleseant'
    elif valence > 0 and arousal > 0:
        return 'intense_pleasent'
    elif valence > 0 and arousal < 0:
        return 'mild_pleasent'
    elif valence < 0 and arousal < 0:
        return 'mild_unpleasent'

def main():
    blockSize = 40
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wtgs = torch.load('TimeSeriesLearningSkip_best_loss.pth.tar')
    model = TimeSeriesLearningSkip()
    model.load_state_dict(wtgs['state_dict'])
    model.to(device)
    frameQt = 28001
    vcap = cv2.VideoCapture("clipe2.mp4")
    #width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    #height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    face_cascade = cv2.CascadeClassifier('cascadeFolder/haarcascade_frontalface_default.xml')
    rois = []
    colors = {}
    face_expressions = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    savedFrame = []
    frameNum = 0
    finalArousal = [[0] * frameQt,[0] * frameQt]
    finalValence = [[0] * frameQt,[0] * frameQt]
    with torch.no_grad():
        model.eval()
        while (vcap.isOpened()):
            print('Examining frame %d' % frameNum)
            sucs, imgv = vcap.read()
            if sucs:
                imgv = cv2.resize(imgv,(640,480))
                faces = detectFace(face_cascade,imgv)
                newROIs = isNewROI(rois,faces)
                for (x,y,w,h) in newROIs:
                    rois.append((x,y,w,h))

                roins = findROI(rois,faces)

                for rncls in set(roins):
                    if rncls not in colors.keys():
                        colors[rncls] = getRandomColors()

                savedFrame.append([imgv,{}])
                for fnum, b in enumerate(faces):
                    if fnum >= len(roins):
                        continue
                    cv2.rectangle(imgv,(b[0],b[1]),(b[0]+b[2],b[1]+b[3]),colors[roins[fnum]],2)
                    savedFrame[-1][1][roins[fnum]] = (b[0],b[1]+b[3]+12)
                    #print("Extraindo face %d" % (roins[fnum]))
                    fImage = imgv[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
                    fImage = cv2.resize(fImage, (100, 100)) - 128
                    fImage = fImage / 128
                    if roins[fnum] not in face_expressions.keys():
                        face_expressions[roins[fnum]] = [[],[]]
                    face_expressions[roins[fnum]][0].append(fImage)
                    face_expressions[roins[fnum]][1].append(frameNum)

                    #cv2.imwrite(os.path.join('frames_face',"roi_" + str(roins[fnum]) + "_frame_"+str(frame_number)+".jpg"),fImage)

                erase = []
                for dface in face_expressions:
                    if len(face_expressions[dface][0]) == blockSize:
                        erase.append(dface)
                        a,v = getArousalValence(face_expressions[dface][0],model)
                        for idx, frame in enumerate(face_expressions[dface][1]):
                            finalValence[dface][idx + (frameNum - blockSize)] = v[idx]
                            finalArousal[dface][idx + (frameNum - blockSize)] = a[idx]
                            cv2.putText(savedFrame[frame][0],getResult(a[idx],v[idx]),savedFrame[frame][1][dface],font,0.5,colors[dface],2,cv2.LINE_AA)
                            cv2.putText(savedFrame[frame][0], "arousal: %f" % a[idx], (savedFrame[frame][1][dface][0],savedFrame[frame][1][dface][1]+20), font, 0.5,
                                        colors[dface], 2, cv2.LINE_AA)
                            cv2.putText(savedFrame[frame][0], "Valence: %f" % v[idx], (savedFrame[frame][1][dface][0],savedFrame[frame][1][dface][1]+35), font, 0.5,
                                        colors[dface], 2, cv2.LINE_AA)
                    elif len(face_expressions[dface][0]) > blockSize:
                        erase.append(dface)

                for d in erase:
                    del face_expressions[d]

                frameNum += 1
                if frameNum == frameQt:
                    break
            else:
                break

    print('Outputing video')
    flatten = lambda t: [item for sublist in t for item in sublist]
    tks = np.arange(min([min(flatten(finalValence)),min(flatten(finalArousal))]),max([max(flatten(finalValence)),max(flatten(finalArousal))]),0.05)
    fig, ax = plt.subplots(2,2)
    canvas = FigureCanvas(fig)
    #ax = fig.gca()
    ax[0][0].set_yticks(tks)
    ax[0][1].set_yticks(tks)
    ax[1][0].set_yticks(tks)
    ax[1][1].set_yticks(tks)
    width, height = fig.get_size_inches() * fig.get_dpi()
    out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (int(width), int(height)*2))
    for idxF, frm in enumerate(savedFrame):
        ax[0][0].set(xlabel='',ylabel='Valence')
        ax[1][0].set(xlabel='Frame',ylabel='Arousal')
        ax[0][0].plot([fnx for fnx in range(frameNum)], finalValence[0][:frameNum])
        ax[0][0].plot([idxF,idxF],[tks.min(),tks.max()])
        ax[1][0].plot([fnx for fnx in range(frameNum)], finalArousal[0][:frameNum])
        ax[1][0].plot([idxF,idxF],[tks.min(),tks.max()])

        ax[0][1].set(xlabel='',ylabel='')
        ax[1][1].set(xlabel='Frame',ylabel='')
        ax[0][1].plot([fnx for fnx in range(frameNum)], finalValence[1][:frameNum])
        ax[0][1].plot([idxF,idxF],[tks.min(),tks.max()])
        ax[1][1].plot([fnx for fnx in range(frameNum)], finalArousal[1][:frameNum])
        ax[1][1].plot([idxF,idxF],[tks.min(),tks.max()])

        canvas.draw()
        im = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        nconc = np.concatenate((frm[0], im))
        out.write(nconc)
        ax[0][0].clear()
        ax[0][1].clear()
        ax[1][0].clear()
        ax[1][1].clear()

    vcap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()