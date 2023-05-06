# -*- coding: utf-8 -*-
import os
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image


class Main:
    inputFiles = os.listdir("Images")
    firstTable = True
    textureLevels = ["сильнотекстурованого", "середньотекстурованого", "слаботекстурованого"]

    def main(self):
        try:
            for inputFile, textureLevel in zip(self.inputFiles, self.textureLevels):
                for index in range(1, 3):
                    self.encode(inputFile, index, textureLevel)
                    self.decoder(inputFile.split('.')[0], index)
                    self.firstTable = False
                self.firstTable = True
        except FileNotFoundError as ex:
            print(ex)

    @classmethod
    def dct_2d(cls, image):
        return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')

    def quantize(self, block, component):
        q = self.load_quantization_table(component)
        return (block / q).round().astype(np.int32)

    def load_quantization_table(self, component):
        if component == 'lum':
            if self.firstTable:
                q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],  # таблиця квантуванння 1
                              [2, 2, 2, 2, 3, 4, 5, 6],
                              [2, 2, 2, 2, 4, 5, 7, 9],
                              [2, 2, 2, 4, 5, 7, 9, 12],
                              [3, 3, 4, 5, 8, 10, 12, 12],
                              [4, 4, 5, 7, 10, 12, 12, 12],
                              [5, 5, 7, 9, 12, 12, 12, 12],
                              [6, 6, 9, 12, 12, 12, 12, 12]])
            else:
                q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # таблиця квантування 2
                              [12, 12, 14, 19, 26, 48, 60, 55],
                              [14, 13, 16, 24, 40, 57, 69, 56],
                              [14, 17, 22, 29, 51, 87, 80, 62],
                              [18, 22, 37, 56, 68, 109, 103, 77],
                              [24, 35, 55, 64, 81, 104, 113, 92],
                              [49, 64, 78, 87, 103, 121, 120, 101],
                              [72, 92, 95, 98, 112, 100, 103, 99]])
        elif component == 'chrom':
            if self.firstTable:
                q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],  # таблица квантувания 1
                              [3, 4, 6, 11, 14, 12, 12, 12],
                              [5, 6, 9, 14, 12, 12, 12, 12],
                              [9, 11, 14, 12, 12, 12, 12, 12],
                              [13, 14, 12, 12, 12, 12, 12, 12],
                              [15, 12, 12, 12, 12, 12, 12, 12],
                              [15, 12, 12, 12, 12, 12, 12, 12],
                              [15, 12, 12, 12, 12, 12, 12, 12]])
            else:
                q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # таблиця квантування 2
                              [18, 21, 26, 66, 99, 99, 99, 99],
                              [24, 26, 56, 99, 99, 99, 99, 99],
                              [47, 66, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99]])
        else:
            raise ValueError(f"компонент має бути 'lum' або 'chrom', але '{component}' знайдено")
        return q

    @classmethod
    def zigzag_points(cls, rows, cols):
        # константи для напрямків
        UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

        # рух точки в різних напрямках
        def move(direction, point):
            return {UP: lambda point: (point[0] - 1, point[1]),
                    DOWN: lambda point: (point[0] + 1, point[1]),
                    LEFT: lambda point: (point[0], point[1] - 1),
                    RIGHT: lambda point: (point[0], point[1] + 1),
                    UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
                    DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
                    }[direction](point)

        def inbounds(point):
            return 0 <= point[0] < rows and 0 <= point[1] < cols

        # ініціалізація початку у верхній лівій клітинці
        point = (0, 0)
        # True коли рух up-right, False коли рух down-left
        move_up = True
        for i in range(rows * cols):
            yield point
            if move_up:
                if inbounds(move(UP_RIGHT, point)):
                    point = move(UP_RIGHT, point)
                else:
                    move_up = False
                    point = move(RIGHT, point) if inbounds(move(RIGHT, point)) else move(DOWN, point)
            else:
                if inbounds(move(DOWN_LEFT, point)):
                    point = move(DOWN_LEFT, point)
                else:
                    move_up = True
                    point = move(DOWN, point) if inbounds(move(DOWN, point)) else move(RIGHT, point)

    def block_to_zigzag(self, block):
        return np.array([block[point] for point in self.zigzag_points(*block.shape)])

    @classmethod
    def bits_required(cls, n):
        n = abs(n)
        result = 0
        while n > 0:
            n >>= 1
            result += 1
        return result

    @classmethod
    def flatten(cls, lst):
        return [item for sublist in lst for item in sublist]

    def run_length_encode(self, arr):
        last_nonzero = max((i for i, elem in enumerate(arr) if elem != 0), default=-1)
        symbols = []
        values = []
        run_length = 0
        for i, elem in enumerate(arr):
            if i > last_nonzero:
                symbols.append((0, 0))
                values.append(self.int_to_binstr(0))
                break
            elif elem == 0 and run_length < 15:
                run_length += 1
            else:
                size = self.bits_required(elem)
                symbols.append((run_length, size))
                values.append(self.int_to_binstr(elem))
                run_length = 0
        return symbols, values

    @classmethod
    def binstr_flip(cls, binstr):
        # перевіряємо, чи є binstr двійковим рядком
        if not set(binstr).issubset('01'):
            raise ValueError("binstr повинен мати лише '0' та '1'")
        return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))

    @classmethod
    def uint_to_binstr(cls, number, size):
        return bin(number)[2:][-size:].zfill(size)

    def int_to_binstr(self, n):
        if n == 0:
            return ''
        binstr = bin(abs(n))[2:]
        return binstr if n > 0 else self.binstr_flip(binstr)

    def write_to_file(self, filepath, dc, ac, blocks_count, tables):
        f = open(filepath, 'w')
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            # 16 біт для 'table_size'
            f.write(self.uint_to_binstr(len(tables[table_name]), 16))
            for key, value in tables[table_name].items():
                if table_name in {'dc_y', 'dc_c'}:
                    # 4 біт для 'category'
                    # 4 біт для 'code_length'
                    # 'code_length' біт для 'huffman_code'
                    f.write(f"{self.uint_to_binstr(key, 4)}{self.uint_to_binstr(len(value), 4)}{value}")
                else:
                    # 4 біт для 'run_length'
                    # 4 біт для 'size'
                    # 8 біт для 'code_length'
                    # 'code_length' bits for 'huffman_code'
                    f.write(f"{self.uint_to_binstr(key[0], 4)}{self.uint_to_binstr(key[1], 4)}"
                            f"{self.uint_to_binstr(len(value), 8)}{value}")

        # 32 біт для 'blocks_count'
        f.write(self.uint_to_binstr(blocks_count, 32))
        for b in range(blocks_count):
            for c in range(3):
                category = self.bits_required(dc[b, c])
                symbols, values = self.run_length_encode(ac[b, :, c])
                dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
                f.write(f"{dc_table[category]}{self.int_to_binstr(dc[b, c])}")
                [f.write(f"{ac_table[tuple(symbols[i])]}{values[i]}") for i in range(len(symbols))]
        f.close()

    def encode(self, inputFile, index, textureLevel):
        output_file = f"Results/{inputFile.split('.')[0]}_table{index}.asf"
        image = Image.open(f"Images/{inputFile}")
        ycbcr = image.convert('YCbCr')
        npmat = np.array(ycbcr, dtype=np.uint8)
        rows, cols = npmat.shape[0], npmat.shape[1]
        if rows % 8 == cols % 8 == 0:
            blocks_count = rows // 8 * cols // 8
        else:
            raise ValueError("ширина і висота зображення мають бути кратними 8")
        dc = np.empty((blocks_count, 3), dtype=np.int32)
        ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
        # block_index = 0
        for i in range(0, rows, 8):
            for j in range(0, cols, 8):
                try:
                    block_index += 1
                except NameError:
                    block_index = 0
                for k in range(3):
                    # split 8x8 block and center the data range on zero
                    # [0, 255] --> [-128, 127]
                    block = npmat[i:i + 8, j:j + 8, k] - 128
                    dct_matrix = self.dct_2d(block)
                    quant_matrix = self.quantize(dct_matrix, 'lum' if k == 0 else 'chrom')
                    zigzag = self.block_to_zigzag(quant_matrix)
                    dc[block_index, k] = zigzag[0]
                    ac[block_index, :, k] = zigzag[1:]
        H_DC_Y = HuffmanTree(np.vectorize(self.bits_required)(dc[:, 0]))
        H_DC_C = HuffmanTree(np.vectorize(self.bits_required)(dc[:, 1:].flat))
        H_AC_Y = HuffmanTree(self.flatten(self.run_length_encode(ac[i, :, 0])
                                          [0] for i in range(blocks_count)))
        H_AC_C = HuffmanTree(self.flatten(self.run_length_encode(ac[i, :, j])
                                          [0] for i in range(blocks_count) for j in [1, 2]))
        tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
                  'ac_y': H_AC_Y.value_to_bitstring_table(),
                  'dc_c': H_DC_C.value_to_bitstring_table(),
                  'ac_c': H_AC_C.value_to_bitstring_table()}
        self.write_to_file(output_file, dc, ac, blocks_count, tables)
        global size_vyhsdnogo
        size_vyhsdnogo = os.path.getsize(f"Images/{inputFile}")
        with open("Results/results_jpeg.txt", "a") as file:
            file.write(f"Дані для {textureLevel} зображення RGB (таблиця {index})\n"
                       f"Розмір вихідного файла: {size_vyhsdnogo} байт\n")

    def read_image_file(self, filepath):
        reader = JPEGFileReader(filepath)
        tables = dict()
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            tables[table_name] = reader.read_dc_table() if 'dc' in table_name else reader.read_ac_table()
        blocks_count = reader.read_blocks_count()
        dc = np.empty((blocks_count, 3), dtype=np.int32)
        ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
        for block_index in range(blocks_count):
            for component in range(3):
                dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if component == 0 else tables['ac_c']
                category = reader.read_huffman_code(dc_table)
                dc[block_index, component] = reader.read_int(category)
                cells_count = 0

                # TODO: спроба зробити читання AC коефіцієнтів краще
                while cells_count < 63:
                    run_length, size = reader.read_huffman_code(ac_table)
                    if (run_length, size) == (0, 0):
                        while cells_count < 63:
                            ac[block_index, cells_count, component] = 0
                            cells_count += 1
                    else:
                        for i in range(run_length):
                            ac[block_index, cells_count, component] = 0
                            cells_count += 1
                        if size == 0:
                            ac[block_index, cells_count, component] = 0
                        else:
                            value = reader.read_int(size)
                            ac[block_index, cells_count, component] = value
                        cells_count += 1
        return dc, ac, tables, blocks_count

    def zigzag_to_block(self, zigzag):
        rows = cols = int(math.sqrt(len(zigzag)))
        if rows * cols != len(zigzag):
            raise ValueError("Довжина зіг-зага повинна бути ідеальним квадратом ")
        block = np.empty((rows, cols), np.int32)
        for i, point in enumerate(self.zigzag_points(rows, cols)):
            block[point] = zigzag[i]
        return block

    def dequantize(self, block, component):
        q = self.load_quantization_table(component)
        return block * q

    @classmethod
    def idct_2d(cls, image):
        return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')

    def decoder(self, fileName, index):
        dc, ac, tables, blocks_count = self.read_image_file(f"Results/{fileName}_table{index}.asf")
        # блок є квадратом 8x8
        block_side = 8
        # висота і ширина зображення рівні
        image_side = int(math.sqrt(blocks_count)) * block_side
        blocks_per_line = image_side // block_side
        npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
        for block_index in range(blocks_count):
            i = block_index // blocks_per_line * block_side
            j = block_index % blocks_per_line * block_side
            for c in range(3):
                zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
                quant_matrix = self.zigzag_to_block(zigzag)
                dct_matrix = self.dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
                block = self.idct_2d(dct_matrix)
                npmat[i:i + 8, j:j + 8, c] = block + 128
        image = Image.fromarray(npmat, 'YCbCr')
        image = image.convert('RGB')
        filename = f"Results/JPEG_{fileName}_result_img_{index}.jpg"
        image.save(filename)
        size_jpeg = os.path.getsize(filename)
        width, height = image.size
        ratio = size_vyhsdnogo / size_jpeg
        with open("Results/results_jpeg.txt", "a") as file:
            file.write(f"Розмір файла JPEG: {size_jpeg} байт\nРозмір зображення JPEG: {width}x{height}\n"
                       f"Коефіцієнт стиснення= {ratio}")


class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32
    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4
    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0
            # старший біт вказує на знак числа
        bin_num = self.__read_str(size)
        return self.__int2(bin_num) if bin_num[0] == '1' else self.__int2(Main().binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for i in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for i in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        prefix = ''
        # TODO: перервати цикл if __read_char не повертає новий символ
        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("розмір повинен бути більшим за 0")
        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    @classmethod
    def __int2(cls, bin_num):
        return int(bin_num, 2)


if __name__ == '__main__':
    Main().main()
