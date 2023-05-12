# -*- coding: utf-8 -*-

from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi
import scipy


class Modulation:

    def __init__(self, ask, psk, fsk1, fsk2):
        self.ask = ask
        self.psk = psk
        self.fsk1 = fsk1
        self.fsk2 = fsk2

    @classmethod
    def plot(cls, x, y, axis_x, axis_y, title, count: int = 1, legend: bool = False):
        fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        [ax.plot(x, y[i], linewidth=1) if count > 1 else ax.plot(x, y, linewidth=1) for i in range(count)]
        ax.set_xlabel(axis_x, fontsize=14)
        ax.set_ylabel(axis_y, fontsize=14)
        ax.legend(['ASK', 'PSK', 'FSK'], loc=2) if legend else None
        plt.title(title, fontsize=14)
        isdir = os.path.isdir('./figures/')
        os.mkdir('./figures/') if not isdir else None
        fig.savefig(f"./figures/{title}.png", dpi=600)
        plt.close(fig)

    @classmethod
    def spectrum(cls, sequence):
        ySpectrum = np.abs(scipy.fft.fftshift(scipy.fft.fft(sequence)))
        xSpectrum = scipy.fft.fftshift(scipy.fft.fftfreq(len(sequence), 1 / len(sequence)))
        return xSpectrum[round(len(xSpectrum) / 2):], ySpectrum[round(len(ySpectrum) / 2):]

    @classmethod
    def create_sequence(cls):
        sequence = np.zeros(1000)
        for i in range(10):
            sequence[i * 100:(i + 1) * 100] = randint(0, 1)
        return sequence

    @classmethod
    def ask_modulation(cls, frequency, sequence):
        sequence_ask = np.zeros(1000)
        for i in range(0, len(sequence)):
            sequence_ask[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)
        return sequence_ask

    @classmethod
    def ask_demodulation(cls, frequency, sequence):
        ask_demodulated_signal = sequence_demodulated = np.zeros(1000)
        threshold = np.ones(1000) * 25
        ask_product = [sequence[i] * cos(2 * pi * frequency * i / 1000) for i in range(0, len(sequence))]
        for i in range(0, 10):
            S = 0
            for t in range(0, 100):
                S += ask_product[t + 100 * (i - 1)]
                ask_demodulated_signal[t + 100 * (i - 1)] = S
        ask_demodulated = 1 / 2 * (np.sign(ask_demodulated_signal - threshold) + 1)
        for i in range(0, 10):
            for t in range(0, 100):
                sequence_demodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]
        return ask_demodulated_signal, sequence_demodulated

    @classmethod
    def psk_modulation(cls, frequency, sequence):
        sequencePsk = [sin(2 * pi * frequency * i / 1000 + sequence[i] * pi + pi) for i in range(len(sequence))]
        return sequencePsk

    @classmethod
    def psk_demodulation(cls, frequency, sequence):
        psk_demodulated_signal = np.zeros(1000)
        threshold = np.ones(1000) * 25
        sequence_demodulated = np.zeros(1000)
        psk_product = [sequence[i] * sin(2 * pi * frequency * i / 1000) for i in range(len(sequence))]
        for i in range(0, 10):
            S = 0
            for t in range(0, 100):
                S = S + psk_product[t + 100 * (i - 1)]
                psk_demodulated_signal[t + 100 * (i - 1)] = S
        psk_demodulated = 1 / 2 * (np.sign(psk_demodulated_signal - threshold) + 1)
        for i in range(0, 10):
            for t in range(0, 100):
                sequence_demodulated[t + 100 * (i - 1)] = psk_demodulated[100 * i - 1]
        return psk_demodulated_signal, sequence_demodulated

    @classmethod
    def fsk_modulation(cls, frequency1, frequency2, sequence):
        sequenceFsk = [sequence[i] * sin(2 * pi * frequency1 * i / 1000) + (abs(sequence[i] - 1)) *
                       sin(2 * pi * frequency2 * i / 1000) for i in range(len(sequence))]
        return sequenceFsk

    @classmethod
    def fsk_demodulation(cls, frequency1, frequency2, sequence):
        fskDemodulatedSignal1 = fskDemodulatedSignal2 = sequenceDemodulated = np.zeros(1000)
        fskProduct1 = [sequence[i] * sin(2 * pi * frequency1 * i / 1000) for i in range(0, len(sequence))]
        fskProduct2 = [sequence[i] * sin(2 * pi * frequency2 * i / 1000) for i in range(0, len(sequence))]
        for i in range(0, 10):
            S1 = 0
            S2 = 0
            for t in range(0, 100):
                S1 = S1 + fskProduct1[t + 100 * (i - 1)]
                fskDemodulatedSignal1[t + 100 * (i - 1)] = S1
                S2 = S2 + fskProduct2[t + 100 * (i - 1)]
                fskDemodulatedSignal2[t + 100 * (i - 1)] = S2
        ask_demodulated = 1 / 2 * (np.sign(fskDemodulatedSignal1 - fskDemodulatedSignal2) + 1)
        for i in range(0, 10):
            for t in range(0, 100):
                sequenceDemodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]
        return fskDemodulatedSignal1, fskDemodulatedSignal2, sequenceDemodulated

    @classmethod
    def create_noise(cls, mean, standard_deviation, length):
        return np.random.normal(mean, standard_deviation, length)

    def noise_stress(self, sequence, sequenceModulated, modulation, frequency):
        errorModulated = []
        sequenceDemodulated = np.zeros(1000)
        for i in range(0, 20):
            p = 0
            for m in range(0, 200):
                noise = self.create_noise(0, 1, 1000)
                sequenceNoise = sequenceModulated + i * noise
                if modulation == "ASK":
                    askDemodulatedSignal, sequenceDemodulated = self.ask_demodulation(frequency[0], sequenceNoise)
                elif modulation == "PSK":
                    pskDemodulatedSignal, sequenceDemodulated = self.psk_demodulation(frequency[0], sequenceNoise)
                elif modulation == "FSK":
                    fskDemodulatedSignal1, fskDemodulatedSignal2, sequenceDemodulated = self.fsk_demodulation(
                        frequency[0], frequency[1], sequenceNoise)
                p += abs(sum(sequence - sequenceDemodulated)) / 1000
            errorModulated += [p / 200]
        return errorModulated

    def main(self):
        sequence = self.create_sequence()
        x = np.arange(len(sequence)) / 1000
        sequenceAsk = self.ask_modulation(self.ask, sequence)
        sequencePsk = self.psk_modulation(self.psk, sequence)
        sequenceFsk = self.fsk_modulation(self.fsk1, self.fsk2, sequence)
        xSpectrumAsk, spectrum_sequence_ask = self.spectrum(sequenceAsk)
        xSpectrumPsk, spectrum_sequence_psk = self.spectrum(sequencePsk)
        xSpectrumFsk, spectrum_sequence_fsk = self.spectrum(sequenceFsk)
        noise = self.create_noise(0, 1, 1000)
        sequenceAskNoise = sequenceAsk + noise
        sequencePskNoise = sequencePsk + noise
        sequenceFskNoise = sequenceFsk + noise
        askDemodulatedSignal, sequenceDemodulatedAsk = self.ask_demodulation(self.ask, sequenceAskNoise)
        pskDemodulatedSignal, sequenceDemodulatedPsk = self.psk_demodulation(self.psk, sequencePskNoise)
        fskDemodulatedSignal1, fskDemodulatedSignal2, sequence_demodulated_fsk = \
            self.fsk_demodulation(self.fsk1, self.fsk2, sequenceFskNoise)
        sequences = [[x, sequence], [x, sequenceAsk], [xSpectrumAsk, spectrum_sequence_ask], [x, sequencePsk],
                     [xSpectrumPsk, spectrum_sequence_psk], [x, sequenceFsk], [xSpectrumFsk, spectrum_sequence_fsk],
                     [x, sequenceAskNoise], [x, sequencePskNoise], [x, sequenceFskNoise], [x, askDemodulatedSignal],
                     [x, sequenceDemodulatedAsk], [x, pskDemodulatedSignal], [x, sequenceDemodulatedPsk],
                     [x, fskDemodulatedSignal1], [x, fskDemodulatedSignal1], [x, sequence_demodulated_fsk]]
        labels = [("Час, c", "Амплітуда сигналу", "Згенерована випадкова послідовність"),
                  ("Час, c", "Амплітуда сигналу", "Амплітудна модуляція"),
                  ("Частота, Гц", "Амплітуда спектру", "Спектр при амплітудній модуляції"),
                  ("Час, c", "Амплітуда сигналу", "Фазова модуляція"),
                  ("Частота, Гц", "Амплітуда спектру", "Спектр при фазовій модуляції"),
                  ("Час, c", "Амплітуда сигналу", "Частотна модуляція"),
                  ("Частота, Гц", "Амплітуда спектру", "Спектр при частотній модуляції"),
                  ("Час, c", "Амплітуда сигналу", "Амплітудна модуляція з шумом"),
                  ("Час, c", "Амплітуда сигналу", "Фазова модуляція з шумом"),
                  ("Час, c", "Амплітуда сигналу", "Частотна модуляція з шумом"),
                  ("Час, c", "Амплітуда сигналу", "Демодульований сигнал з амплітудною модуляцією"),
                  ("Час, c", "Амплітуда сигналу", "Демодульована послідовність з амплітудною модуляцією"),
                  ("Час, c", "Амплітуда сигналу", "Демодульований сигнал з фазовою модуляцією"),
                  ("Час, c", "Амплітуда сигналу", "Демодульована послідовність з фазовою модуляцією"),
                  ("Час, c", "Амплітуда сигналу", "Демодульований сигнал 1 з частотною модуляцією"),
                  ("Час, c", "Амплітуда сигналу", "Демодульований сигнал 2 з частотною модуляцією")]
        for key, value in zip(sequences, labels):
            self.plot(key[0], key[1], value[0], value[1], value[2])
        errors = (self.noise_stress(sequence, sequenceAsk, "ASK", [self.ask]),
                  self.noise_stress(sequence, sequencePsk, "PSK", [self.psk]),
                  self.noise_stress(sequence, sequenceFsk, "FSK", [self.fsk1, self.fsk2]))
        self.plot(np.arange(0, 20), errors, 'Діапазон змін шуму', 'Ймовірність помилки',
                  'Оцінка завадостійкості трьох видів модуляції', count=3, legend=True)


if __name__ == '__main__':
    Modulation(20, 40, 40, 20).main()
