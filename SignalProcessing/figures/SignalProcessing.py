import numpy
from scipy import signal, fft
import matplotlib.pyplot as plt


class Main:

    def __init__(self):
        self.n = 500
        self.Fs = 1000
        self.FMax = 3
        self.Dt = [2, 4, 8, 16]
        self.FFilter = 10

    def main(self):
        randomNormal = numpy.random.normal(0, 10, self.n)
        x = numpy.arange(self.n) / self.Fs
        w = self.FMax / (self.Fs / 2)
        parametersFilter = signal.butter(3, w, "low", output="sos")
        y = signal.sosfiltfilt(parametersFilter, randomNormal)
        frequencyReadings = fft.fftfreq(self.n, 1 / self.n)
        frequencyReadingsCenter = fft.fftshift(frequencyReadings)
        discreteSignals = []
        discreteSpectrums = []
        vars2 = []
        SNRList = []
        for Dt in self.Dt:
            discreteSignal = numpy.zeros(self.n)
            for i in range(0, round(self.n / Dt)):
                discreteSignal[i * Dt] = y[i * Dt]
            discreteSpectrum = fft.fft(discreteSignal)
            discreteSpectrum = numpy.abs(fft.fftshift(discreteSpectrum))
            discreteSignals.append(discreteSignal)
            discreteSpectrums.append(discreteSpectrum)
            E1 = discreteSignal - y
            var1 = numpy.var(y)
            var2 = numpy.var(E1)
            vars2.append(var2)
            SNR = var1 / var2
            SNRList.append(SNR)
        w = self.FFilter / (self.Fs / 2)
        parametersFilter = signal.butter(3, w, 'low', output='sos')
        y = signal.sosfiltfilt(parametersFilter, discreteSignals)
        self.subplot(x, discreteSignals, "Час (секунди)", "Амплітуда сигналу",
                     "Сигнал з кроком дисркетизації Dt = (2, 4, 8, 16)")
        self.subplot(frequencyReadingsCenter, discreteSpectrums, "Частота(Гц)", "Амплітуда спектру",
                     "Спектри сигналів з кроком дисркетизації Dt = (2, 4, 8, 16)")
        self.subplot(x, y, "Час (секунди)", "Амплітуда сигналу",
                     "Відновлені аналогові сигнали з кроком дисркетизації Dt = (2, 4, 8, 16)")
        self.plot(self.Dt, vars2, "Крок дискретизації", "Дисперсія",
                  "Залежність дисперсії від кроку дискретизації")
        self.plot(self.Dt, SNRList, "Крок дискретизації", "ССШ",
                  "Залежність співвідношення сигнал-шум від кроку дискретизації")

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
    def plot(x, y, xLabel, yLabel, title):
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


if __name__ == '__main__':
    Main().main()
