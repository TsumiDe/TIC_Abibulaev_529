# -*- coding: utf-8 -*-

import random


class HammingCode:
    chunkLength = 8
    assert not chunkLength % 8, "Довжина блоку має бути кратна 8"
    checkBits = [i for i in range(1, chunkLength + 1) if not i & (i - 1)]

    def main(self):
        with open("results_hamming.txt", "w", encoding="utf-8") as file:
            file.close()
        with open("sequence.txt", 'r') as file:
            originalSequences = [i[15:-1] for i in file.readlines()[1::2]]
        for sequence in originalSequences:
            source_bin, encoded = self.encode(sequence[:10])
            decoded = self.decode(encoded)
            encodedWithError = self.getSetErrors(encoded)
            diffIndexList = self.getDiffIndexList(encoded, encodedWithError)
            decodedWithError = self.decode(encodedWithError, fixErrors=False)
            decodedWithoutError = self.decode(encodedWithError)
            with open("results_hamming.txt", "a", encoding="utf-8") as file:
                file.write(f"//////////////////////////////////////////////////\n"
                           f"Оригінальна послідовність: {sequence[:10]}\n"
                           f"Оригінальна послідовність в бітах: {source_bin}\n"
                           f"Розмір оригінальної послідовності: {str(source_bin.__len__())}\n"
                           f"Довжина блоку кодування: {self.chunkLength}\nПозиція контрольних біт: {self.checkBits}\n"
                           f"Відносна надмірність коду: {self.checkBits.__len__() / self.chunkLength}\n"
                           f"----------Кодування----------\nЗакодовані дані: {encoded}\n"
                           f"Розмір закодованих даних: {encoded.__len__()}\n---------Декодування---------\n"
                           f"Декодовані дані: {decoded}\nРозмір декодованих даних: {decoded.__len__() * 16}\n"
                           f"-------Внесення помилки-------\nЗакодовані дані з помилками: {encodedWithError}\n"
                           f"Кількість помилок: {diffIndexList.__len__()}\nІндекси помилок: {diffIndexList}\n"
                           f"Декодовані дані без виправлення помилки: {decodedWithError}\n"
                           f"-----Виправлення помилки-----\n"
                           f"Декодовані дані з виправленням помилки: {decodedWithoutError}\n"
                           f"//////////////////////////////////////////////////\n\n")

    # Кодування даних
    def encode(self, source):
        textBin = self.getCharsToBin(source)
        result = ''.join([self.getSetCheckBits(chunk_bin) for chunk_bin in self.getChunkIterator(textBin)])
        return textBin, result

    # Декодування даних
    def decode(self, encoded, fixErrors=True):
        decoded_value = ''
        getChunkIterator = self.getChunkIterator(encoded, self.chunkLength + self.checkBits.__len__())
        fixedEncodedList = [self.getCheckAndFixError(encodedChunk) if fixErrors else encodedChunk
                            for encodedChunk in getChunkIterator]
        cleanChunkList = ''.join([self.getExcludeCheckBits(encodedChunk) for encodedChunk in fixedEncodedList])
        for j in range(0, cleanChunkList.__len__(), 16):
            cleanChunk = cleanChunkList[j:j + 16]
            for clean_char in [cleanChunk[i:i + 16] for i in range(cleanChunk.__len__()) if not i % 16]:
                decoded_value += chr(int(clean_char, 2))
        return decoded_value

    # Перетворення символів в бінарний формат
    def getCharsToBin(self, chars):
        assert not chars.__len__() * 8 % self.chunkLength, \
            "Довжина кодових даних повинна бути кратною довжині блоку кодування"
        return ''.join([bin(ord(c))[2:].zfill(16) for c in chars])

    # Поблоковий вивід бінарних даних
    @classmethod
    def getChunkIterator(cls, text_bin, chunkSize=chunkLength):
        for i in range(len(text_bin)):
            if not i % chunkSize:
                yield text_bin[i:i + chunkSize]

    # Отримання інформації про контрольні біти з бінарного блоку даних при кодуванні
    def getCheckBitsData(self, valueBin):
        checkBitsCountMap = {k: 0 for k in self.checkBits}
        for index, value in enumerate(valueBin, 1):
            if int(value):
                binCharList = list(bin(index)[2:].zfill(8))
                binCharList.reverse()
                for degree in [2 ** int(i) for i, value in enumerate(binCharList) if int(value)]:
                    checkBitsCountMap[degree] += 1
        checkBitsValueMap = {check_bit: 0 if not count % 2 else 1 for check_bit, count in checkBitsCountMap.items()}
        return checkBitsValueMap

    # Додавання порожніх контрольних біт в бінарні дані
    def getSetEmptyCheckBits(self, valueBin):
        for bit in self.checkBits:
            valueBin = valueBin[:bit - 1] + '0' + valueBin[bit - 1:]
        return valueBin

    # Встановлення значень контрольних біт
    def getSetCheckBits(self, valueBin):
        valueBin = self.getSetEmptyCheckBits(valueBin)
        checkBitsData = self.getCheckBitsData(valueBin)
        for checkBit, bitValue in checkBitsData.items():
            valueBin = f"{valueBin[:checkBit - 1]}{bitValue}{valueBin[checkBit:]}"
        return valueBin

    # Отримання інформації про контрольні біти з бінарного блоку даних при декодуванні
    def getCheckBits(self, valueBin):
        checkBits = {index: int(value) for index, value in enumerate(valueBin, 1) if index in self.checkBits}
        return checkBits

    # Видалити контрольні біти
    def getExcludeCheckBits(self, valueBin):
        cleanValueBin = ''.join([char_bin for index, char_bin in enumerate(list(valueBin), 1)
                                 if index not in self.checkBits])
        return cleanValueBin

    # Додати помилку до бінарної послідовності
    def getSetErrors(self, encoded):
        result = ''
        for chunk in self.getChunkIterator(encoded, self.chunkLength + self.checkBits.__len__()):
            numBit = random.randint(1, len(chunk))
            result += f"{chunk[:numBit - 1]}{int(chunk[numBit - 1]) ^ 1}{chunk[numBit:]}"
        return result

    # Пошук та виправлення помилок при передачі
    def getCheckAndFixError(self, encodedChunk):
        checkBitsEncoded = self.getCheckBits(encodedChunk)
        checkItem = self.getSetCheckBits(self.getExcludeCheckBits(encodedChunk))
        checkBits = self.getCheckBits(checkItem)
        if checkBitsEncoded != checkBits:
            invalidBits = [checkBitEncoded for checkBitEncoded, value in checkBitsEncoded.items()
                           if checkBits[checkBitEncoded] != value]
            numBit = sum(invalidBits)
            encodedChunk = f"{encodedChunk[:numBit - 1]}{int(encodedChunk[numBit - 1]) ^ 1}{encodedChunk[numBit:]}"
        return encodedChunk

    # Список індексів позицій в яких було допущено помилки
    @classmethod
    def getDiffIndexList(cls, valueBin1, valueBin2):
        diffIndexList = [index for index, charBinItems in enumerate(zip(list(valueBin1), list(valueBin2)), 1)
                         if charBinItems[0] != charBinItems[1]]
        return diffIndexList


if __name__ == '__main__':
    HammingCode().main()
