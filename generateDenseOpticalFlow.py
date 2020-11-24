import numpy as np
import cv2
from DatasetClasses.FEDatasets import getCASME2BlockData

def runDenseOpFlow(files):
    oriImg = cv2.imread(files[0])
    prvs = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(oriImg).astype(np.float32)
    hsv[..., 1] = 255
    for idx, f in enumerate(files[1:]):
        nextFrame = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, nextFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("teste_%d.png" % (idx),cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        prvs = nextFrame

    hsv = cv2.normalize(hsv, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    labels, fcs = getCASME2BlockData('CASME2_formated','train',5)
    hsvImage = runDenseOpFlow(fcs[0])
    cv2.imwrite('teste.jpg',hsvImage)