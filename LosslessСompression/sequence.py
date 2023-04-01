import random
from string import ascii_lowercase, digits
import math
import collections
from matplotlib import pyplot as plt


class Main:

    def __init__(self, firstValue, secondValue, shuffle: bool, typeNumber: str, p, sequenceNumber, tag):
        self.sequenceN = 100
        self.firstValue = firstValue
        self.secondValue = secondValue
        self.shuffle = shuffle
        self.typeNumber = typeNumber
        self.p = p
        self.originalSequences = []
        self.sequenceNumber = sequenceNumber
        self.tag = tag

    def main(self, results):
        originalSequence = ''
        if self.typeNumber == "First":
            originalSequence = [i for i in list(self.firstValue)] + [self.secondValue for _ in
                                                                     range(self.sequenceN - len(self.firstValue))]
        elif self.typeNumber == "Second":
            originalSequence = [self.firstValue[i % len(self.firstValue)] for i in range(self.sequenceN)]
        elif self.typeNumber == "Third":
            originalSequence = [random.choice(self.firstValue) for _ in range(self.sequenceN)]
        elif self.typeNumber == "Fourth":
            originalSequence = [random.choice(self.firstValue) for _ in range(int(self.p[0] * self.sequenceN))] + \
                               [random.choice(self.secondValue) for _ in range(int(self.p[1] * self.sequenceN))]
        elif self.typeNumber == "Fifth":
            originalSequence = list(self.firstValue) * 20
        if self.shuffle:
            random.shuffle(originalSequence)
        originalSequence = ''.join(originalSequence)
        sequenceAlphabetSize = len(set(originalSequence))
        originalSequenceSize = len(originalSequence)
        self.originalSequences.append(originalSequence)
        counts = collections.Counter(originalSequence)
        probability = {symbol: count / self.sequenceN for symbol, count in counts.items()}
        probability_str = ', '.join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])
        mean_probability = sum(probability.values()) / len(probability)
        equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in
                    probability.values())
        uniformity = "рівна" if equal else "нерівна"
        entropy = -sum(p * math.log2(p) for p in probability.values())
        sourceExcess = 1 - entropy / math.log2(sequenceAlphabetSize) if sequenceAlphabetSize > 1 else 1
        results.append([sequenceAlphabetSize, round(entropy, 2), round(sourceExcess, 2), uniformity])
        self.saveTxt(name="sequence.txt", originalSequence=originalSequence)
        self.saveTxt(name="resultsSequence.txt", originalSequence=originalSequence,
                     sequenceAlphabetSize=sequenceAlphabetSize, originalSequenceSize=originalSequenceSize,
                     probability=probability_str, mean_probability=mean_probability, uniformity=uniformity,
                     entropy=entropy, sourceExcess=sourceExcess)

    def saveTxt(self, name, originalSequence, sequenceAlphabetSize=None, originalSequenceSize=None, probability=None,
                mean_probability=None, uniformity=None, entropy=None, sourceExcess=None):
        with open(name, self.tag, encoding="windows-1251") as file:
            if name == "resultsSequence.txt":
                file.write(f"originalSequence {self.sequenceNumber}:\nПослідовність: {originalSequence}\n"
                           f"Розмір послідовності: {originalSequenceSize} byte\nРозмір алфавіту: {sequenceAlphabetSize}\n"
                           f"Ймовірності появи символів: {probability}\nСереднє арифметине ймовірностей: {mean_probability}\n"
                           f"Ймовірність розподілу символів: {uniformity}\nЕнтропія: {entropy}\n"
                           f"Надмірність джерела: {sourceExcess}\n")
            else:
                file.write(f"originalSequence {self.sequenceNumber}:\nПослідовність: {originalSequence}\n")


def createTable(results, headers, row):
    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig("Характеристики сформованих послідовностей")


def main():
    keys = [{"firstValue": '1', "secondValue": '0', "shuffle": True, "type": "First", "p": None, "tag": 'w'},
            {"firstValue": "Абібулаєв", "secondValue": '0', "shuffle": False, "type": "First", "p": None, "tag": 'a'},
            {"firstValue": "Абібулаєв", "secondValue": '0', "shuffle": True, "type": "First", "p": None, "tag": 'a'},
            {"firstValue": "Абібулаєв529", "secondValue": None, "shuffle": False, "type": "Second", "p": None,
             "tag": 'a'},
            {"firstValue": "Аб529", "secondValue": None, "shuffle": True, "type": "Fifth", "p": None, "tag": 'a'},
            {"firstValue": "Аб", "secondValue": "529", "shuffle": True, "type": "Fourth", "p": [0.7, 0.3], "tag": 'a'},
            {"firstValue": ascii_lowercase + digits, "secondValue": None, "shuffle": True, "type": "Third", "p": None,
             "tag": 'a'},
            {"firstValue": '1', "secondValue": None, "shuffle": False, "type": "Second", "p": None, "tag": 'a'}]
    number = 1
    results = []
    for key in keys:
        Main(firstValue=key.get("firstValue"), secondValue=key.get("secondValue"), shuffle=key.get("shuffle"),
             typeNumber=key.get("type"), p=key.get('p'), sequenceNumber=number, tag=key.get("tag")).main(
            results=results)
        number += 1
    headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
    row = [f"Послідовність {n}" for n in range(1, number)]
    createTable(results=results, headers=headers, row=row)


if __name__ == '__main__':
    main()
