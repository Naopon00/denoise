from socket import timeout
from turtle import up
import streamlit as st
import scipy.signal as sp
import wave as wave
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from pydub import AudioSegment

st.set_option('deprecation.showPyplotGlobalUse', False)
def monoral(upload_files):
    sound = AudioSegment.from_wav(upload_files)
    sound = sound.set_channels(1)
    sound.export(upload_files, format = "wav")
    return upload_files

def saveSoundData(name, data, framerate):
    data = data.astype(np.int16)
    wave_out = wave.open(name, "w")
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(framerate)
    wave_out.writeframes(data)
    wave_out.close()
    st.audio(name)
    with open(name, "rb") as file:
        btn = st.download_button(
            label="Download File",
            data = file,
            file_name = name,
            mime = "audio/wav"
        )

def plotSpectrogram(data,framerate):
    fig = plt.figure(figsize = (10,4))
    spectrum, freqs, t, im = plt.specgram(data, 
                NFFT = 512, noverlap = 256, Fs = framerate, cmap = "jet")

    fig.colorbar(im).set_label('Intensity[dB]')
    plt.xlabel('Time[sec]')
    plt.ylabel('Freque ncy[Hz]')
    figure = plt.show()
    st.pyplot(figure)

def denoise(upload_files):
    wav = wave.open(upload_files)
    mix_signal = wav.readframes(wav.getnframes())
    mix_signal = np.frombuffer(mix_signal, dtype = np.int16)

    Fs = wav.getframerate()
    n_speech = wav.getnframes()

    f, t, stft_data = sp.stft(mix_signal, fs = wav.getframerate(), window = "hann", nperseg = 512, noverlap = 256)
    amp = np.abs(stft_data)
    phase = stft_data / np.maximum(amp, 1.e-20)

    n_noise_only = 40000
    n_noise_only_frame = np.sum(t < (n_noise_only / Fs))

    p = 1.0
    alpha = 110.0
    micro = 10000.0

    nlk = np.mean(np.power(amp, p)[:,:n_noise_only_frame], axis = 1, keepdims = True)
    noise_amp = np.power(nlk, 1./p)

    eps = 0.01 * np.power(amp, 2)
    slk = np.maximum(np.power(amp, 2) - alpha * np.power(nlk, 2), eps)

    rlk = np.power(slk, 2) / (np.power(slk, 2) + micro * np.power(nlk, 2))
    slkht = rlk * stft_data
    
    t, processed_data_post = sp.istft(slkht, fs = wav.getframerate(), window = "hann", nperseg = 512, noverlap = 256)
    saveSoundData("processed_data.wav", processed_data_post, Fs)
    st.write("### 実行前のスペクトログラム")
    plotSpectrogram(stft_data, Fs)
    st.write("### 雑音除去後のスペクトログラム")
    plotSpectrogram(processed_data_post, Fs)

def main():
    st.write("""
    # 雑音除去アプリ
    音声ファイルをアップロードして値を設定し、実行すると雑音を除去したファイルが作成され、ダウンロードすることができます。
    """)
    # janken()

    upload_files = st.file_uploader("")
    st.audio(upload_files)

    if upload_files is not None:
        running = st.button("実行する")
        upload_files = monoral(upload_files)
        if running:
            denoise(upload_files)

# def janken():
#     st.sidebar.write("じゃんけん")
#     hands = ['グー', 'チョキ', 'パー']
#     option = st.sidebar.selectbox(
#         'じゃんけんの手',
#         hands
#     )
#     myhand = 0
#     for l in range(3):
#         if option == hands[l]:
#             myhand = l

#     handbutton = st.sidebar.button('じゃんけんをする')
#     if handbutton:
#         hand = random.randint(0, 2)
#         if myhand == hand:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write('あいこです')
#         elif myhand == 0 and hand == 1:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの勝ちです")
#         elif myhand == 0 and hand == 2:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの負けです")
#         elif myhand == 1 and hand == 0:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの負けです")
#         elif myhand == 1 and hand == 2:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの勝ちです")
#         elif myhand == 2 and hand == 0:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの勝ちです")
#         elif myhand == 2 and hand == 1:
#             st.sidebar.write("CPUの手：", hands[hand])
#             st.sidebar.write("あなたの負けです")
            
main()