import sys
import numpy as np
from BoundingBoxes import *


def getBoundingBoxes():
    import glob
    import os
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath, 'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    allBoundingBoxes = BoundingBoxes()
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, None,
                BBType.GroundTruth,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    folderDet = os.path.join(currentPath, 'detections')
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()

    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, None,
                BBType.Detected,
                confidence,
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

boundingboxes = getBoundingBoxes()

def _getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def _getUnionAreas(boxA, boxB, interArea=None):
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)



def _iou(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou


def GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.5):
    ret = []  # list containing metrics (precision, recall, average precision) of each class
    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    groundTruths = []
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = []
    # Get all classes
    classes = []
    # Loop through all bounding boxes and separate them into GTs and detections
    for bb in boundingboxes.getBoundingBoxes():
        # [imageName, class, confidence, (bb coordinates XYX2Y2)]
        if bb.getBBType() == BBType.GroundTruth:
            groundTruths.append([
                bb.getImageName(),
                bb.getClassId(), 1,
                bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            ])
        else:
            detections.append([
                bb.getImageName(),
                bb.getClassId(),
                bb.getConfidence(),
                bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            ])
        # get class
        if bb.getClassId() not in classes:
            classes.append(bb.getClassId())
    classes = sorted(classes)
    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if (d[1] == c and d[2] > 0.5)]
        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [g]

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}

        # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = gts[dects[d][0]] if dects[d][0] in gts else []
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                iou = _iou(dects[d][3], gt[j][3])
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.sum(FP)
        acc_TP = np.sum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        # add class result in the dictionary to be returned
        r = {
            'class': c,
            'num': npos,
            'precision': prec,
            'recall': rec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP)
        }
        ret.append(r)
    return ret

metricsPerClass = GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.3,)
print("Average precision values per class:\n")

for mc in metricsPerClass:
    print('class:', mc['class'])
    print('num:', mc['num'])
    print('tp:', mc['total TP'])
    print('fp:', mc['total FP'])
    print('recall:', mc['recall'])
    print('precision:', mc['precision'])
    print('====================================================')