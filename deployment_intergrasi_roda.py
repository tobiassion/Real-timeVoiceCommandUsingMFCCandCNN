import librosa
import numpy as np
import pyaudio
import wave
import os
import math
import warnings
import struct
import threading
import serial
import tensorflow as tf
import time
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

SHORT_NORMALIZE = (1.0/32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
swidth = 2
CHUNK = 1024
TimeoutSignal = ((RATE / CHUNK * 1)+2)
TEMPORARY_WAVE_FILENAME = "tempxiv.wav"
SAVED_MODEL_PATH = "C:/Users/Lenovo/Documents/Education/Skripsi/Coding/Kumpulan Model/Riset/glrs_3c_2f1fcl.h5"
STRING = ''
silence = True
Threshold = 50
audio = pyaudio.PyAudio()

ArduinoSerial=serial.Serial ('com4',9600,timeout=0.1)

isStop = False
STOPPING = False

def checkStop():
    global lastStop, STOPPING, t1, t2
    if lastStop == -1:
        print("Updating lastStop")
        lastStop = time.time()
        return
    if time.time() - lastStop > 3:
        print("FITUR SUARA DIMATIKAN")
        lastStop = -1
        ArduinoSerial.write(bytes("1", 'utf-8'))
        STOPPING = True
        t1.join()
        t2.join()
        exit()

def resetStop():
    global lastStop
    # print("RESETTING LASTSTOP")
    lastStop = -1
    return

class _Keyword_Spotting_Service:

    model = None
    _mapping = [
        "Go",
        "Left",
        "Right",
        "Stop"
    ]
    _instance = None

    def predict(self, file_path):
        start_time_MFCC = time.time()
        MFCCs = self.preprocess(file_path)
        waktu_komputasi_MFCC = time.time() - start_time_MFCC
        print("--- Waktu Ekstraksi Fitur MFCC : %s detik ---" % (waktu_komputasi_MFCC))

        # Penamahan 1 layer di awal dan 1 layer di akhir yang membuat bantuk array menjadi [sample, timesample, banyaknya MFCC, channel]
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # Prediksi menggunkan model
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        print(predictions)
        predicted_keyword = self._mapping[predicted_index]  
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load file audio
        signal, sample_rate = librosa.load(file_path)

        # Melihat panjang sinyal, dan merubah panjangan sinyal menjadi 22050 (di potong atau di padding)
        length = librosa.get_duration(signal)
        print('signal before padding', signal.shape)
        if length != 2:
            signal = librosa.util.fix_length(signal, 22050)
            print("Signal after padding", signal.shape)

        if len(signal) >= RATE:
            # ensure consistency of the length of the signal
            signal = signal[:RATE]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T

def Keyword_Spotting_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance

def rms(frame):
    count = len(frame)/swidth
    format = "%dh"%(count)
    # short is 16 bit int
    shorts = struct.unpack( format, frame )
    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms
    rms = math.pow(sum_squares/count,0.5)
    return rms * 1000

def recording(lastblock, stream):
    global FORMAT, CHUNK, CHANNELS, RATE, TEMPORARY_WAVE_FILENAME
    try:
        arr = []
        arr.append(lastblock)
        print ("recording...")
        for i in range(0, int(TimeoutSignal)):
            data = stream.read(CHUNK)
            arr.append(data)

        print ("Finish recordings")

        waveFile = wave.open(TEMPORARY_WAVE_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(arr))
        waveFile.close()
        recognise()
        del stream
    except Exception as e:
        print (e)
        raise

def recognise():
    global STRING
    global silence
    start_time_prediksi = time.time()
    kss = Keyword_Spotting_Service()
    keyword = kss.predict(TEMPORARY_WAVE_FILENAME)
    print("Perintah: " + keyword)
    waktu_komputasi_CNN = time.time() - start_time_prediksi
    print("--- Waktu Prediksi CNN : %s detik ---" % (waktu_komputasi_CNN))
    if(keyword == 'Go'): 
        STRING = "0"
    elif(keyword == 'Stop'): 
        STRING = "1"
    elif(keyword == 'Right'): 
        STRING = "2"
    elif(keyword == 'Left'): 
        STRING = "3"
    silence = True

def listen():
    global silence, STOPPING
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
    
    while True:
        while silence:
            try:
                input = stream.read(CHUNK)
            except:
                print("error")
                continue
            rms_value = rms(input)
            if (rms_value > Threshold):
                silence=False
                LastBlock=input
                recording(LastBlock, stream)
            if STOPPING:
                return
        if STOPPING:
            return

def spam():
    global STRING, STOPPING
    while True: 
        if STRING=='1':
            checkStop()
        else:
            resetStop()
        ArduinoSerial.write(bytes(STRING, 'utf-8'))
        PWM = ArduinoSerial.readline().decode().strip()
        print(PWM) 
        if STOPPING:
            return
            
t2 = threading.Thread(target=spam, args=())
t1 = threading.Thread(target=listen, args=())

t1.start()
t2.start()

t1.join()
t2.join()


 
