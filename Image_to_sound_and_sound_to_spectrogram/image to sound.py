import librosa
import numpy as np
import wave, struct, math
from PIL import Image
from scipy import ndimage
from librosa import display
from scipy.io import wavfile
import matplotlib.pyplot as plt

image = Image.open("black_hole.jpeg")
image = image.convert("L")


def converting(size, image):
    image_np = np.array(image)
    image_np = np.flip(image_np, axis=0)
    image_np -= np.min(image_np)
    image_np = image_np / np.max(image_np)
    if size[0] == 0:
        size = image_np.shape[0], size[1]
    if size[0] == 1:
        size = size[0], image_np.shape[1]
    resampling_factor = size[0] / image_np.shape[0], size[1] / image_np.shape[1]
    if resampling_factor[0] == 0:
        resampling_factor = 1, resampling_factor[1]
    if resampling_factor[1] == 0:
        resampling_factor = resampling_factor[0], 1
    image_np = ndimage.zoom(image_np, resampling_factor, order=0)
    return image_np


def creatingSound(file, output="sound.wav", duration=5.0, sampleRate=44100, intensityFactor=1, min_freq=0,
                  max_freq=22000, invert=False):
    wave_file = wave.open(output, 'w')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(sampleRate)

    max_frame = int(sampleRate * duration)
    max_intensity = 32767
    stepsize = 400
    steppingSpectrum = int((max_freq - min_freq) / stepsize)

    image_np = converting(size=(steppingSpectrum, max_frame), image=file)
    if invert:
        image_np = 1 - image_np
    image_np *= intensityFactor
    image_np *= max_intensity
    for frame in range(max_frame):
        signalValue, count = 0, 0
        for step in range(steppingSpectrum):
            intensity = image_np[step, frame]
            if intensity < 0.1 * intensityFactor:
                continue
            currentFreq = (step * stepsize) + min_freq
            nextFreq = ((step + 1) * stepsize) + min_freq
            if nextFreq - min_freq > max_freq:
                nextFreq = max_freq
            for freq in range(currentFreq, nextFreq, 1000):
                signalValue += intensity * math.cos(freq * 2 * math.pi * float(frame) / float(sampleRate))
                count += 1
        if count == 0: count = 1
        signalValue /= count

        data = struct.pack('<h', int(signalValue))
        wave_file.writeframesraw(data)

    wave_file.writeframes(''.encode())
    wave_file.close()


def plotSpectrogram(file="sound.wav"):
    sample_rate, X = wavfile.read(file)
    print(X, max(X), min(X))
    try:
        if X.shape[1] > 1:
            y = []

            for x in range(len(X)):
                y.append(int(np.mean(X[x])))
            y = np.array(y)
            plt.specgram(y, Fs=sample_rate, xextent=(0, 60))
    except:
        # plt.plot(X)
        plt.specgram(X, Fs=sample_rate, xextent=(0, 60))
    print("File: ", file)
    print("Sample rate (Hz): ", sample_rate)
    plt.show()


creatingSound(image)
plotSpectrogram()
