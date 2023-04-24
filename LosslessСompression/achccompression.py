# -*- coding: utf-8 -*-
import collections
import math
from matplotlib import pyplot as plt


class Main:

    def __init__(self, readFileName, writeFileName):
        self.originalSequences: list = self.readTXT()
        self.sequenceN: int = 100
        self.readFileName: str = readFileName
        self.writeFileName: str = writeFileName
        self.results: list = []
        self.mode = 'w'
        self.ntCode = ['\n', '\t']

    def main(self):
        for sequence in self.originalSequences:
            sequenceLength = sequence.__len__()
            uniqueChars = set(sequence)
            sequenceAlphabetSize = set(uniqueChars).__len__()
            counts = collections.Counter(sequence)
            probability = {symbol: count / sequenceLength for symbol, count in counts.items()}
            entropy = -sum(p * math.log2(p) for p in probability.values())
            encodedDataAC, encodedSequenceAC = self.encodeAC(uniqueChars=uniqueChars, probabilities=probability,
                                                             sequenceAlphabetSize=sequenceAlphabetSize,
                                                             sequence=sequence)
            bpsAC = len(encodedSequenceAC) / sequenceLength
            decodedSequenceAC = self.decodeAC(encodedDataAC=encodedDataAC, sequenceLength=sequenceLength)
            encodedDataCH, encodedSequenceCH = self.encodeCH(alphabet=list(uniqueChars), probabilities=probability, sequence=sequence)
            bpsCH = len(encodedSequenceCH) / sequenceLength
            decodedSequenceCH = self.decodeCH(encode=list(encodedDataCH[0]), symbolCode=encodedDataCH[1])
            self.writeTXT(value=f"//////////////////////////////////////////////////\n"
                                f"Оригінальна послідовність: {sequence}\nЕнтропія: {entropy}\n\n"
                                f"__________Арифметичне кодування__________\n"
                                f"Дані закодованої AC послідовності: {encodedDataAC}\n"
                                f"Закодована AC послідовність: {encodedSequenceAC}\n"
                                f"Значення bps при кодуванні AC: {bpsAC}\n"
                                f"Декодована AC послідовність: {decodedSequenceAC}\n\n"
                                f"__________Кодування Хаффмана__________\n"
                                f"Алфавіт   Код символу\n"
                                f"{''.join([f'{self.ntCode[1]}{symbol}{self.ntCode[1] * 2}{code}{self.ntCode[0]}' for symbol, code in encodedDataCH[1]])}"
                                f"Дані закодованої CH послідовності: {encodedDataCH}\n"
                                f"Закодована CH послідовність: {encodedSequenceCH}\n"
                                f"Значення bps при кодуванні CH: {bpsCH}\n"
                                f"Декодована CH послідовність: {decodedSequenceCH}\n"
                                f"//////////////////////////////////////////////////\n\n")
            self.mode = 'a'
            self.results.append([round(entropy, 2), round(bpsAC, 1), round(bpsCH, 1)])
        self.table()

    def encodeAC(self, uniqueChars, probabilities, sequenceAlphabetSize, sequence):
        alphabet = list(uniqueChars)
        probability = [probabilities[symbol] for symbol in alphabet]
        unity = self.unity(sequenceAlphabetSize=sequenceAlphabetSize, probability=probability, alphabet=alphabet)
        for i in range(len(sequence) - 1):
            for j in range(len(unity)):
                if sequence[i] == unity[j][0]:
                    probability_low, probability_high = unity[j][1], unity[j][2]
                    diff = probability_high - probability_low
                    for k in range(len(unity)):
                        unity[k][1], unity[k][2] = probability_low, probability[k] * diff + probability_low
                        probability_low = unity[k][2]
                    break
        low, high = [[unity[i][1], unity[i][2]] for i in range(len(unity)) if unity[i][0] == sequence[-1]][0]
        point = (low + high) / 2
        sizeCod = math.ceil(math.log((1 / (high - low)), 2) + 1)
        binCode = self.floatBin(point=point, sizeCod=sizeCod)
        return [point, sequenceAlphabetSize, alphabet, probability], binCode

    def decodeAC(self, encodedDataAC, sequenceLength):
        point, sequenceAlphabetSize, alphabet, probability = encodedDataAC
        unity = self.unity(sequenceAlphabetSize=sequenceAlphabetSize, probability=probability, alphabet=alphabet)
        decodedSequence = ""
        for i in range(sequenceLength):
            for j in range(unity.__len__()):
                if unity[j][1] < point < unity[j][2]:
                    probabilityLow, probabilityHigh = unity[j][1], unity[j][2]
                    diff = probabilityHigh - probabilityLow
                    decodedSequence += unity[j][0]
                    for k in range(unity.__len__()):
                        unity[k][1], unity[k][2] = probabilityLow, probability[k] * diff + probabilityLow
                        probabilityLow = unity[k][2]
                    break
        return decodedSequence

    def encodeCH(self, alphabet, probabilities, sequence):
        probability = [probabilities[symbol] for symbol in alphabet]
        tree = []
        symbolCode = []
        final = [[alphabet[i], probability[i]] for i in range(alphabet.__len__())]
        final.sort(key=lambda x: x[1])
        if 1 in probability and len(set(probability)) == 1:
            symbolCode = [[alphabet[i], "1" * i + "0"] for i in range(alphabet.__len__())]
            encode = "".join([symbolCode[alphabet.index(c)][1] for c in sequence])
        else:
            left = final[0::2]
            right = final[1::2]
            for i in range(len(final) - 1):
                i = 0
                left = final[i]
                final.pop(i)
                right = final[i]
                final.pop(i)
                tot = left[1] + right[1]
                tree.append([left[0], right[0]])
                final.append([left[0] + right[0], tot])
                final.sort(key=lambda x: x[1])
            tree.reverse()
            alphabet.sort()
            for i in range(len(alphabet)):
                code = ""
                for j in range(len(tree)):
                    if alphabet[i] in tree[j][0]:
                        code += '0'
                        if alphabet[i] == tree[j][0]:
                            break
                    else:
                        code += '1'
                        if alphabet[i] == tree[j][1]:
                            break
                symbolCode.append([alphabet[i], code])
            encode = ""
            for c in sequence:
                encode += [symbolCode[i][1] for i in range(len(alphabet)) if symbolCode[i][0] == c][0]
        return [encode, symbolCode], encode

    def decodeCH(self, encode, symbolCode):
        sequence = ""
        count = 0
        flag = 0
        for i in range(len(encode)):
            for j in range(len(symbolCode)):
                if encode[i] == symbolCode[j][1]:
                    sequence += str(symbolCode[j][0])
                    flag = 1
            if flag == 1:
                flag = 0
            else:
                count = count + 1
                if count == len(encode):
                    break
                else:
                    encode.insert(i + 1, (encode[i] + encode[i + 1]).__str__())
                    encode.pop(i + 2)
        return sequence

    @staticmethod
    def floatBin(point, sizeCod):
        binaryCode = ""
        for x in range(sizeCod):
            point *= 2
            binaryCode, x, point = [binaryCode + '1', int(point), point - x] if point > 1 \
                else([binaryCode + '0', x, point] if point < 1 else [binaryCode + '1', x, point])
        return binaryCode

    @staticmethod
    def unity(sequenceAlphabetSize, probability, alphabet):
        unity = []
        probabilityRange = 0.0
        for i in range(sequenceAlphabetSize):
            l = probabilityRange
            probabilityRange += probability[i]
            u = probabilityRange
            unity.append([alphabet[i], l, u])
        return unity

    def readTXT(self):
        with open("sequence.txt", 'r') as file:
            return [i[15:26] for i in file.readlines()[1::2]]

    def writeTXT(self, value):
        with open(self.writeFileName, self.mode) as file:
            file.write(value)
            file.close()

    def table(self):
        fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
        headers = ["Ентропія", "bps AC", "bps CH"]
        row = [f"Послідовність {i}" for i in range(1, 9)]
        ax.axis('off')
        table = ax.table(cellText=self.results, colLabels=headers, rowLabels=row,
                         loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        fig.savefig("Результати стиснення методами AC та CH")


if __name__ == '__main__':
    Main(readFileName="sequence.txt", writeFileName="results_AC_CH.txt").main()
