def waveform():
    import librosa
    import matplotlib.pyplot as plt
    import librosa.display

    array, sampling_rate = librosa.load(librosa.ex("trumpet"))

    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.show()

def spectrum():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    array, sampling_rate = librosa.load(librosa.ex("trumpet"))

    # 截取一段
    dft_input = array[:4096]

    # 计算DFT
    window = np.hanning(len(dft_input))  # 设置一个hanning窗口，让原始信号更加平滑
    windowed_input = dft_input * window
    dft = np.fft.rfft(windowed_input)  # 进行傅里叶变换

    # 获得响度的频谱，分贝格式, 也就是纵坐标
    amplitude = np.abs(dft)  # dft是复数形式，用 abs 计算每个频率成分的幅度，也就是信号在该频率上的强度
    amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

    # 获得频率区间, 也就是横坐标
    frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

    # 作图
    plt.figure().set_figwidth(12)
    plt.plot(frequency, amplitude_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.xscale("log")
    plt.show()

def spectrogram():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt

    array, sampling_rate = librosa.load(librosa.ex("trumpet"))
    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.show()

def mel_spectrogram():
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt

    array, sampling_rate = librosa.load(librosa.ex("trumpet"))
    S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    import os
    os.environ['LIBROSA_DATA_DIR'] = '../cache/librosa/'
    # waveform()
    # spectrum()
    # spectrogram()
    mel_spectrogram()