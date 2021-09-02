
import librosa, librosa.display
import csv
import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd

file_name = "./a.wav"
wf = wave.open(file_name, mode='rb')
sr = wf.getframerate()
fr = wf.getnframes()

# fft
threshold = 300
dt = 1/sr
t = np.arange(0, fr/sr, dt)

wf.rewind() # ポインタを先頭に戻す
buf = wf.readframes(-1) # 全部読み込む

# 2なら16bit，4なら32bitごとに10進数化
if wf.getsampwidth() == 2:
    wav = np.frombuffer(buf, dtype='int16')
elif wf.getsampwidth() == 4:
    wav = np.frombuffer(buf, dtype='int32')
'''
window = 5 # 移動平均の範囲
w = np.ones(window)/window

x = np.convolve(wav, w, mode='same')

'''
x = np.fft.fft(wav)
x_abs = np.abs(x) # 複素数を絶対値に変換
x_abs = x_abs / fr * 2 # 振幅の調整
x[x_abs < threshold] = 0

#fq = np.linspace(0, 1.0/dt, N) # 周波数の調整

x = np.fft.ifft(x)
x = x.real

x = x[:44032]
wav = wav[:44032]

print(wav)
print(wf.getsampwidth())
print(x)
print(t)

fig, ax = plt.subplots(figsize=(14.0, 6.0))
ax.plot(t, x, label='IFFT')
ax.plot(t, wav, alpha=0.3, label='wave')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.grid()
plt.legend()
plt.title("Inverse Fast Fourier Transform")

fig, ax = plt.subplots(figsize=(14.0, 6.0))
ax.plot(t, x, label='IFFT')
ax.plot(t, wav, alpha=0.3, label='wave')
ax.set_xlim(0, 0.06)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.grid()
plt.legend()
plt.title("enlarged view")
fig.savefig("img.png")


# to grasshopper setting

csvdataamp = x[:1000]
csvdatatime = t[:1000]

def calc_10000(n):
    return n * 10000

def calc_001(n):
    return n * 0.01

a = map(calc_001, csvdataamp)
t = map(calc_10000, csvdatatime)
z = [1] * 1000

df = pd.DataFrame({'amplitude':a, 'time':t, 'z':z})
df.T
df.reset_index(drop=True)
df.to_csv('./data.csv')
