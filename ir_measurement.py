###############################################################################
#                                                                             #
#    This program measures the acoustic impulse response of a room using      #
#    logarithmic sine sweep                                                   #
#                                                                             #
#    Technical School of Madrid (UPM)                                         #
#    Author:    Javier Nistal Hurle                                           #
#    Supervisor:    Dr. Lino Garcia Morales                                   #
#    University: Technical School of Madrid (UPM)                             #
###############################################################################
import sys

import pyaudio
import wave
from array import array
from struct import pack
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import chirp
from pylab import plot, figure, show, subplot, title
from test.test_isinstance import AbstractClass
from abc import ABCMeta, abstractmethod
from _pyio import __metaclass__
########## CONSTANTES ##############
nombre_IR=r'ir_atico.wav';
RUTA=r''
SWEEP=r'barrido.wav'
CHUNK = 512
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 15
INPUT_DEVICE=0
OUTPUT_DEVICE=4
FORMAT = pyaudio.paInt16
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
FRAME_MAX_VALUE = 2 ** 15 - 1


def save_file(data, path):

    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

class ExcitationSignal:
    __metaclass__ = ABCMeta
    def __init__(self, type='sweep', sample_rate=44100, duration=5):
        self.type = type
        self.sample_rate = sample_rate
        self.duration = duration
        
    @abstractmethod
    def create(self):
        pass
    
    @abstractmethod
    def get_ir(self):
        pass


class SineSweep(ExcitationSignal):
    def __init__(self, method='logarithmic'):
        super(SineSweep, self).__init__()
        self.method = method
        
    def create(self):
        
        t = np.linspace(
                0, 
                self.duration, 
                self.sample_rate * self.duration - 1
            )
        sweep = chirp(t, 20, self.duration, 20000, self.method) * 32767
        
        return sweep.astype(np.int16)

    @staticmethod
    def get_ir(sweep, sweep_response):

        y = fft(sweep_response, len(sweep_response))
        x = fft(sweep, len(sweep_response))

        ir = ifft(np.divide(y,x)).real

        return np.around([x*32768 for x in ir])

class IRMeasurement(object):
    def __init__(self, n_ch=1, sr=44100, frame_size=CHUNK):
        self.n_ch = n_ch
        self.sr = sr
        self.frame_size = frame_size
        
    def generate(self, signal):
            
        zero_padd = np.zeros(self.frame_size - np.mod(len(signal), self.frame_size))
        signal = np.append(signal, zero_padd)
        
        for i in range(0, len(signal)/self.frame_size):
            yield signal[i*self.frame_size:(i+1)*self.frame_size].astype(np.int16).tostring()
            
    def measure(self, signal):
        data_sweep = array('h')
        recording = array('h')

        print("* recording")
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                channels=self.n_ch,
                rate=self.sr,
                input=True,
                output=True,
                input_device_index=INPUT_DEVICE,
                output_device_index=OUTPUT_DEVICE,
                frames_per_buffer=CHUNK)

        for frame in self.generate(signal):
            
            stream.write(frame, self.frame_size)
            recording.fromstring(stream.read(self.frame_size))
  
        stream.stop_stream()
        stream.close()
        p.terminate()
        return recording

class IRwindow(object):
    
    @staticmethod
    def lundeby(ir):
        
        mean_db = array('f')
        mean = array('f')
        eixo_tempo = array('f')
        find = array('f')
    
        energy = np.power(ir, 2)
        maxenergy = max(energy) + sys.float_info.min
        t = int(np.floor(len(energy) / RATE / 0.01))
        v = np.floor(len(energy) / t)

        rms_dB_tail = 10 * np.log10(np.mean(np.divide(
                        energy[int(round(0.9 * len(energy))) : len(energy)], maxenergy
                    )))
    
        for x in range(1, t+1):
            mean.append(np.mean(energy[int((x-1)*v):int((x*v)-1)]))
            eixo_tempo.append(np.ceil(v/2)+((x-1)*v))
    
        mean_db = map(lambda a: 10*np.log10(np.divide(a, maxenergy)), mean)

        r = 0
        for i in range(0, len(mean_db)):
            if (mean_db[i] > rms_dB_tail + 10) and (r < i):
                r = i
        
        for i in range(0,r):
            if mean_db[i] < rms_dB_tail + 10:
                find.append(i)
    
        if not find:
            r = 10
        else:
            r = int(min(find))
        if r < 10:
            r = 10
            
        xi = eixo_tempo[0 : r-1]
        A = np.vstack([xi, np.ones(len(xi))]).T
        y = mean_db[0 : r-1]
        b, m = np.linalg.lstsq(A, y)[0]         
        cruzamento = (rms_dB_tail - m) / b

        if rms_dB_tail > -20:
            #Relacao sinal ruido insuficiente
            ponto=len(energy);
        else:
            erro = 1.0;
            INTMAX = 50.0;
            vezes = 1.0;
            while ((erro > 0.0001) and (vezes <= INTMAX)):
                #Calculo de nuevos intervalos,
                #con p pasos por cada 10dB
                r = t = v = n = mean = eixo_tempo = None
                mean = array('f')
                
                eixo_tempo = array('f');          ' %numero de passos por decada'
                p = 5;
                
                if b is 0:
                    b = sys.float_info.min
    
                delta = abs(10 / b);
                v = np.floor(delta / p);        '%intervalo para obtencao de mean'
    
                if abs(cruzamento) == np.inf:
                    cruzamento = len(energy)
                
                t = int(np.floor(len(energy[0:int(round(cruzamento-delta)-1)])/v))
                if t < 2 or not t:
                    t=2
                
                for n in range(1,t+1):
                    mean.append(np.mean(energy[int(((n-1)*v)):int((n*v)-1)]))
                    eixo_tempo.append(np.ceil(v/2)+((n-1)*v));
    
                meandB = map(lambda x: 10 * np.log10(np.divide(x, max(energy))), mean)
    
                xi = m = b = noise = rms_dB_tail = None
    
                xi = eixo_tempo
                A = np.vstack([xi, np.ones(len(xi))]).T
                y = meandB
                b, m = np.linalg.lstsq(A, y)[0]
    
    
                noise = energy[int(round(cruzamento+delta)):len(energy)]
    
                if (len(noise) < round(.1*len(energy))):
                    noise = energy[int(round(.9*len(energy))):len(energy)-1]
    
    
                rms_dB = 10*np.log10(np.divide(np.mean(noise), max(energy)))
                erro = abs(cruzamento - (rms_dB - m) / b)/cruzamento;
                cruzamento = round((rms_dB - m) / b);
                vezes = vezes + 1;

        if cruzamento > len(energy):
            ponto = len(energy)
        else:
            ponto = cruzamento
            
        return ir[:int(abs(ponto))]

if __name__=='__main__':
    
 
    sweep = SineSweep().create()

    sweep_response = IRMeasurement().measure(sweep)
    ir = SineSweep.get_ir(sweep, sweep_response)
    ir = IRwindow.lundeby(ir[np.argmax(ir):])
    plot(ir)
    show()
    
    print