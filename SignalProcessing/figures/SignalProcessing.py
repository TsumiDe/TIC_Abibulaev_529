import numpy
from scipy import signal, fft
import matplotlib.pyplot as plt


class Main:

    def __init__(self):
        self.n = 500
        self.Fs = 100
        self.FMax = 3

    def main(self):
        randomNormal = numpy.random.normal(0, 10, self.n)
        x = numpy.arange(self.n) / self.Fs
        w = self.FMax / (self.Fs / 2)
        parametersFilter = signal.butter(3, w, "low", output="sos")
        y = signal.sosfiltfilt(parametersFilter, randomNormal)
        self.plot(x, y, "Час (секунди)", "Амплітуда сигналу", "Сигнал")

        spectrum = fft.fft(y)
        spectrumCenter = numpy.abs(fft.fftshift(spectrum))
        frequencyReadings = fft.fftfreq(self.n, 1 / self.n)
        frequencyReadingsCenter = fft.fftshift(frequencyReadings)
        self.plot(frequencyReadingsCenter, spectrumCenter, "Частота(Гц)", "Амплітуда спектру", "Спектр сигналу")

    def plot(self, x, y, xLabel, yLabel, title):
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
