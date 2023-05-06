# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random


class Main:
    outFilesDir = "Results"
    names = ("First frame", "Second frame", "Difference between frame", "Prediction frame", "Residual frame",
             "Restore frame")

    def __init__(self, fileName):
        self.fileName = fileName

    def main(self):
        frame = random.randint(0, 3000)
        firstFrame, secondFrame = self.getFrames(self.fileName, frame, frame + 1)
        self.mainCode(firstFrame, secondFrame, saveOutput=True)

    def mainCode(self, anchorFrame, targetFrame, saveOutput: bool = False, blockSize=16):
        bitsAnchor = []
        bitsDiff = []
        bitsPredicted = []
        h, w, ch = anchorFrame.shape
        diffFrameRGB = np.zeros((h, w, ch))
        predictedFrameRGB = np.zeros((h, w, ch))
        residualFrameRGB = np.zeros((h, w, ch))
        restoreFrameRGB = np.zeros((h, w, ch))
        for i in range(0, 3):
            anchorFrame_c = anchorFrame[:, :, i]
            targetFrame_c = targetFrame[:, :, i]
            diffFrame = cv2.absdiff(anchorFrame_c, targetFrame_c)
            predictedFrame = self.blockSearchBody(anchorFrame_c, targetFrame_c, blockSize)
            residualFrame = self.getResidual(targetFrame_c, predictedFrame)
            reconstructTargetFrame = self.getReconstructTarget(residualFrame, predictedFrame)
            bitsAnchor += [self.getBitsPerPixel(anchorFrame_c)]
            bitsDiff += [self.getBitsPerPixel(diffFrame)]
            bitsPredicted += [self.getBitsPerPixel(residualFrame)]
            diffFrameRGB[:, :, i] = diffFrame
            predictedFrameRGB[:, :, i] = predictedFrame
            residualFrameRGB[:, :, i] = residualFrame
            restoreFrameRGB[:, :, i] = reconstructTargetFrame
        if saveOutput:
            for name, value in zip(self.names, (anchorFrame, targetFrame, diffFrameRGB, predictedFrameRGB,
                                                residualFrameRGB, restoreFrameRGB)):
                self.imWriter(name=name, value=anchorFrame)
        barWidth = 0.25
        plt.subplots(figsize=(12, 8))
        P1 = [sum(bitsAnchor), bitsAnchor[0], bitsAnchor[1], bitsAnchor[2]]
        Diff = [sum(bitsDiff), bitsDiff[0], bitsDiff[1], bitsDiff[2]]
        Mpeg = [sum(bitsPredicted), bitsPredicted[0], bitsPredicted[1], bitsPredicted[2]]
        br1 = np.arange(len(P1))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        print(br4)
        plt.bar(br1, P1, color='r', width=barWidth, edgecolor='grey', label='Розмір для початкового кадру')
        plt.bar(br2, Diff, color='g', width=barWidth, edgecolor='grey', label='Розмір для різниці між кадрами')
        plt.bar(br3, Mpeg, color='b', width=barWidth, edgecolor='grey', label='Розмір для різниці з компенсацією рухів')
        plt.title(f"Ступінь стиснення = {round(sum(bitsAnchor) / sum(bitsPredicted), 2)}", fontweight="bold",
                  fontsize=15)
        plt.ylabel("Біт на піксель", fontweight="bold", fontsize=15)
        plt.xticks([r + barWidth for r in range(len(P1))],
                   ["Біт/Піксель RGB", "Біт/Піксель R", "Біт/Піксель G", "Біт/Піксель B"])
        plt.legend()
        plt.savefig(f"{self.outFilesDir}/Гістограма кількості біт на піксель для різних варіантів кодування.png",
                    dpi=600)

    # функція для зчитування сусідніх кадрів з відео
    @classmethod
    def getFrames(cls, fileName, firstFrame, secondFrame):
        cap = cv2.VideoCapture(fileName)
        cap.set(cv2.CAP_PROP_POS_FRAMES, firstFrame - 1)
        res1, fr1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, secondFrame - 1)
        res2, fr2 = cap.read()
        cap.release()
        return fr1, fr2

    # функція для розрахунку кількості біт на піксель
    @classmethod
    def getBitsPerPixel(cls, im):
        h, w = im.shape
        im_list = im.tolist()
        bits = sum([math.log2(abs(pixel) + 1) for row in im_list for pixel in row])
        return bits / (h * w)

    # функція для відновлення кадру(декодування)
    @classmethod
    def getReconstructTarget(cls, residual, predicted):
        return np.add(residual, predicted)

    # функція для пошуку загальної різниці між кадрами
    @classmethod
    def getResidual(cls, target, predicted):
        return np.subtract(target, predicted)

    # функція, яка об’єднує усі процедури пошуку блоків(кодування)
    def blockSearchBody(self, anchor, target, blockSize, searchArea=7):
        h, w = anchor.shape
        hSegments, wSegments = self.segmentImage(anchor, blockSize)
        predicted = np.ones((h, w)) * 255
        blockCount = 0
        for y in range(0, int(hSegments * blockSize), blockSize):
            for x in range(0, int(wSegments * blockSize), blockSize):
                blockCount += 1
                targetBlock = target[y:y + blockSize, x:x + blockSize]
                anchorSearchArea = self.getAnchorSearchArea(x, y, anchor, blockSize, searchArea)
                anchorBlock = self.getBestMatch(targetBlock, anchorSearchArea, blockSize)
                predicted[y:y + blockSize, x:x + blockSize] = anchorBlock
        assert blockCount == int(hSegments * wSegments)
        return predicted

    # функція, в якій буде виконуватись розрахунок кількості сегментів кадру
    @classmethod
    def segmentImage(cls, anchor, blockSize=16):
        h, w = anchor.shape
        hSegments, wSegments = map(int, (h / blockSize, w / blockSize))
        return hSegments, wSegments

    # функція для розрахунку зони кадру, в якому відбуватиметься пошук схожого блоку кадру
    def getAnchorSearchArea(self, x, y, anchor, blockSize, searchArea):
        h, w = anchor.shape
        cx, cy = self.getCenter(x, y, blockSize)
        sx, sy = map(lambda c: max(0, c[0] - int(blockSize / 2) - searchArea), ((cx,), (cy,)))
        anchorSearch = anchor[sy:min(sy + searchArea * 2 + blockSize, h), sx:min(sx + searchArea * 2 + blockSize, w)]
        return anchorSearch

    # функція для розрахунку центру блоку зображення
    @classmethod
    def getCenter(cls, x, y, blockSize):
        return map(int, (x + blockSize / 2, y + blockSize / 2))

    # функція для самого пошуку максимально схожого блоку зображення
    def getBestMatch(self, tBlock, aSearch, blockSize):
        step = 4
        ah, aw = aSearch.shape
        acy, acx = int(ah / 2), int(aw / 2)
        minMAD = float("+inf")
        minP = None
        while step >= 1:
            pointList = [(acx, acy), (acx + step, acy), (acx, acy + step), (acx + step, acy + step), (acx - step, acy),
                         (acx, acy - step), (acx - step, acy - step), (acx + step, acy - step),
                         (acx - step, acy + step)]
            for p in range(len(pointList)):
                aBlock = self.getBlockZone(pointList[p], aSearch, tBlock, blockSize)
                MAD = self.getMAD(tBlock, aBlock)
                minMAD, minP = (MAD, pointList[p]) if MAD < minMAD else (minMAD, minP)
            step = int(step / 2)
        px, py = minP
        px, py = px - int(blockSize / 2), py - int(blockSize / 2)
        px, py = max(0, px), max(0, py)
        matchBlock = aSearch[py:py + blockSize, px:px + blockSize]
        return matchBlock

    # функція для вибору зони кадру для блоку
    @classmethod
    def getBlockZone(cls, p, aSearch, tBlock, blockSize):
        px, py = p
        px, py = px - int(blockSize / 2), py - int(blockSize / 2)
        px, py = max(0, px), max(0, py)
        aBlock = aSearch[py:py + blockSize, px:px + blockSize]
        try:
            assert aBlock.shape == tBlock.shape
        except Exception as e:
            print(e)
        return aBlock

    # функція для розрахунку параметру, який відображає ступінь схожості блоків
    @classmethod
    def getMAD(cls, tBlock, aBlock):
        return np.sum(np.abs(np.subtract(tBlock, aBlock))) / (tBlock.shape[0] * tBlock.shape[1])

    def imWriter(self, name, value):
        cv2.imwrite(f"{self.outFilesDir}/{name}.png", value)


if __name__ == '__main__':
    Main(fileName="sample4.avi").main()
