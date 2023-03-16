import numpy
from scipy import signal
import matplotlib.pyplot as plt


class Main:

    def __init__(self):
        self.n = 500
        self.Fs = 1000
        self.FMax = 3
        self.FFilter = 10
        self.M = [4, 16, 64, 256]

    def main(self):
        randomNormal = numpy.random.normal(0, 10, self.n)
        x = numpy.arange(self.n) / self.Fs
        w = self.FMax / (self.Fs / 2)
        parametersFilter = signal.butter(3, w, "low", output="sos")
        y = signal.sosfiltfilt(parametersFilter, randomNormal)
        quantizeSignals = []
        dispersions = []
        SNRList = []
        for M in self.M:
            bits = []
            delta = (numpy.max(y) - numpy.min(y)) / (M - 1)
            quantizeSignal = delta * numpy.round(y / delta)
            quantizeLevels = numpy.arange(numpy.min(quantizeSignal), numpy.max(quantizeSignal) + 1, delta)
            quantizeBit = numpy.arange(0, M)
            quantizeBit = [format(bits, '0' + str(int(numpy.log(M) / numpy.log(2))) + 'b') for bits in quantizeBit]
            quantizeSignals.append(quantizeSignal)
            quantizeTable = numpy.c_[quantizeLevels[:M], quantizeBit[:M]]
            self.plotFirst(M, quantizeTable, "Значення сигналу", "Кодова послідовність",
                           f"Таблиця квантування для {M} рівнів")
            for signal_value in quantizeSignal:
                for index, value in enumerate(quantizeLevels[:M]):
                    if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                        bits.append(quantizeBit[index])
                        break
            bits = [int(item) for item in list(''.join(bits))]
            self.plotThird(x=numpy.arange(0, len(bits)), y=bits, xLabel="Біти", yLabel="Амплітуда сигналу",
                           title=f"Кодова послідовність сигналу при кількості рівнів квантування {M}")
            E = quantizeSignal - y
            dispersion = numpy.var(E)
            SNR = numpy.var(y) / dispersion
            dispersions.append(dispersion)
            SNRList.append(SNR)
        self.subplot(x=x, y=quantizeSignals, xLabel="Час (секунди)", yLabel="Амплітуда сигналу",
                     title="Цифрові сигнали з рівнями квантування (4, 16, 64, 256)")
        self.plotSecond(x=self.M, y=dispersions, xLabel="Кількість рівнів квантування", yLabel="Дисперсія",
                        title="Залежність дисперсії від кількості рівнів квантування")
        self.plotSecond(x=self.M, y=SNRList, xLabel="Кількість рівнів квантування", yLabel="ССШ",
                        title="Залежність співвідношення сигнал-шум від кількості рівнів квантування")

    @staticmethod
    def subplot(x, y, xLabel, yLabel, title):
        try:
            fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
            s = 0
            for i in range(0, 2):
                for j in range(0, 2):
                    ax[i][j].plot(x, y[s], linewidth=1)
                    s += 1
            fig.supxlabel(xLabel, fontsize=14)
            fig.supylabel(yLabel, fontsize=14)
            fig.suptitle(title, fontsize=14)
            fig.savefig(f"{title}.png", dpi=600)
            plt.show()
        except Exception as ex:
            print(ex)

    @staticmethod
    def plotFirst(M, quantizeTable, colLabelsFirst, colLabelsSecond, title):
        try:
            fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
            table = ax.table(cellText=quantizeTable, colLabels=[colLabelsFirst, colLabelsSecond],
                             loc="center")
            table.set_fontsize(14)
            table.scale(1, 2)
            ax.axis('off')
            fig.savefig(f"{title}.png", dpi=600)
            plt.show()
        except Exception as ex:
            print(ex)

    @staticmethod
    def plotSecond(x, y, xLabel, yLabel, title):
        try:
            fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
            ax.plot(x, y, linewidth=1)
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            plt.title(title, fontsize=14)
            fig.savefig(f"{title}.png", dpi=600)
            plt.show()
        except Exception as ex:
            print(ex)

    @staticmethod
    def plotThird(x, y, xLabel, yLabel, title):
        try:
            fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
            ax.step(x, y, linewidth=0.1)
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            plt.title(title, fontsize=14)
            fig.savefig(f"{title}.png", dpi=600)
            plt.show()
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    Main().main()
