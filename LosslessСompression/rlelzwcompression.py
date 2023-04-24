# -*- coding: utf-8 -*-
import collections
import math
from matplotlib import pyplot as plt


class Main:

    def __init__(self):
        self.sequenceN = 100
        self.compressionRatioRLE = None
        self.compressionRatioLZW = None

    def main(self):
        results = []
        for sequence in self.readTxt():
            counts = collections.Counter(sequence)
            probability = {symbol: count / self.sequenceN for symbol, count in counts.items()}
            entropy = -sum(p * math.log2(p) for p in probability.values())
            self.writeTXT(text=f"Оригінальная послідовність: {str(sequence)}\n"
                               f"Розмір оригінальної послідовності: {str(len(sequence) * 16)} bytes\n"
                               f"Ентропія: {str(entropy)}\n")
            self.decodeRLE(self.encodeRLE(sequence))
            self.decodeLZW(self.encodeLZW(sequence))
            results.append([round(entropy, 2), self.compressionRatioRLE, self.compressionRatioLZW])
        self.table(results=results)

    @staticmethod
    def readTxt():
        with open("sequence.txt", 'r') as file:
            originalSequences = [i[15:-1] for i in file.readlines()[1::2]]
            return originalSequences

    def encodeRLE(self, sequence):
        score: int = 1
        sequenceEncodeRLE = []
        for index, item in enumerate(sequence):
            if index == 0:
                continue
            elif item == sequence[index - 1]:
                score += 1
            else:
                sequenceEncodeRLE.append((sequence[index - 1], score))
                score = 1
        sequenceEncodeRLE.append((sequence[len(sequence) - 1], score))
        encoded = "".join([f"{item[1]}{item[0]}" for item in sequenceEncodeRLE])
        self.compressionRatioRLE = round((len(sequence) / len(encoded)), 2)
        self.compressionRatioRLE = '-' if self.compressionRatioRLE < 1 else self.compressionRatioRLE
        return encoded, sequenceEncodeRLE

    def decodeRLE(self, sequenceEncodeRLE):
        sequenceDecodeRLE = ''.join([symbol[0] * symbol[-1] for symbol in sequenceEncodeRLE[1]])
        self.writeTXT(text=f"_______Кодування RLE_______\n"
                           f"Закодована RLE послідовність: {sequenceEncodeRLE[0]}\n"
                           f"Розмір закодованої RLE послідовності: {str(len(sequenceEncodeRLE[0]) * 16)} bits\n"
                           f"Коефіцієнт стиснення RLE: {str(self.compressionRatioRLE)}\n"
                           f"Декодована RLE послідовність: {sequenceDecodeRLE}\n"
                           f"Розмір декодованої RLE послідовності: {str(len(sequenceDecodeRLE) * 16)} bits\n"
                           f"_______Кодування LZW_______\n"
                           f"____Поетапне кодування____\n")

    def encodeLZW(self, sequence):
        dictionary = {}
        for i in range(65536):
            dictionary[chr(i)] = i
        result = []
        size = 0
        current = ""
        for c in sequence:
            new_str = current + c
            if new_str in dictionary:
                current = new_str
            else:
                result.append(dictionary[current])
                dictionary[new_str] = len(dictionary)
                element_bits = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
                self.writeTXT(text=f"Code: {dictionary[current]}, Element: {current}, bits: {element_bits}\n")
                current = c
                size += element_bits
        last = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
        size += last
        self.writeTXT(text=f"Code: {dictionary[current]}, Element: {current}, bits: {last}\n"
                           f"____________________________________\n")
        result.append(dictionary[current])
        self.compressionRatioLZW = round((len(sequence) * 16 / size), 2)
        self.compressionRatioLZW = '-' if self.compressionRatioLZW < 1 else self.compressionRatioLZW
        self.writeTXT(text=f"Закодована LZW послідовність: {''.join(map(str, result))} \n"
                           f"Розмір закодованої LZW послідовності: {size} bits \n"
                           f"Коефіціент стиснення LZW: {self.compressionRatioLZW}\n")
        return result, size

    def decodeLZW(self, sequenceEncodeLZW):
        dictionary = {}
        for i in range(65536):
            dictionary[i] = chr(i)
        result = ""
        previous = None
        for code in sequenceEncodeLZW[0]:
            if code in dictionary:
                current = dictionary[code]
                result += current
                if previous is not None:
                    dictionary[len(dictionary)] = current[0] + previous
                previous = current
            else:
                current = previous + previous[0]
                result += current
                dictionary[len(dictionary)] = current
                previous = current
        self.writeTXT(text=f"Декодована LZW послідовність: {result} \n"
                           f"Розмір декодованої LZW послідовності: {len(result) * 16}bits \n")

    @staticmethod
    def writeTXT(text):
        with open("results_rle_lzw.txt", 'a') as file:
            file.write(text)

    @staticmethod
    def table(results):
        fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
        headers = ['Ентропія', 'КС RLE', 'КС LZW']
        row = [f"Послідовність {i}" for i in range(1, 9)]
        ax.axis('off')
        table = ax.table(cellText=results, colLabels=headers, rowLabels=row,
                         loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        fig.savefig("Результати стиснення методами RLE та LZW")


if __name__ == '__main__':
    Main().main()
