###############################################################################
#                                                                             #
#    This program measures the acoustic impulse response of a room using      #
#    logarithmic sine sweep                                                   #
#                                                                             #
#    Technical School of Madrid (UPM)                                         #
#    Author:    Javier Nistal Hurle                                           #
#    Supervisor:    Dr. Lino Garcia Morales                                   #
#                                                                             #
###############################################################################

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
import sys
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
    def generate(self):
        pass
    
    @abstractmethod
    def get_ir(self):
        pass


class SineSweep(ExcitationSignal):
    def __init__(self, method='logarithmic'):
        super(SineSweep, self).__init__()
        self.method = method
        
    def generate(self):
        sweep = chirp(
            np.linspace(
                0, 
                self.duration, 
                self.sample_rate * self.duration - 1
            ), 
            20, 
            self.duration, 
            20000, 
            self.method
        ) * 32767
                              
        return sweep

    @staticmethod
    def get_ir(sweep, sweep_response):

        y = fft(sweep_response, len(sweep_response))
        x = fft(sweep, len(sweep_response))

        ir = ifft(np.divide(y,x)).real

        return np.around([x*32768 for x in ir])

class IRMeasurement(object):
    def __init__(self, signal, n_ch=1, sr=44100):
        self.n_ch = n_ch
        self.sr = sr
        self.signal = signal

    def measure(self):
        data_sweep = array('h')
        data_rec = array('h')

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

        data = self.signal.astype(np.int16).tostring()

        wf = wave.open(RUTA+SWEEP, 'r') #Devuelve un string
        dataOUT = wf.readframes(CHUNK)
        i = 0
        print self.signal
#         while dataOUT != '':
#             stream.write(data, CHUNK)
#             data_chunk = stream.read(CHUNK)
# 
#             data_sweep.fromstring(data)
#             dataOUT = wf.readframes(CHUNK)
#             data_rec.fromstring(data_chunk)
#             i += 1
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        return data_rec


def lundeby(ir):
    mean_db = array('f')
    mean = array('f')
    eixo_tempo = array('f')
    find = array('f')

    energy = np.power(ir,2)
    maxenergy = max(energy)

    rms_dB = 10*np.log10(np.mean(np.divide(
                 energy[round(0.9*len(energy)):len(energy)],max(energy)
                 )))

    t = int(np.floor(len(energy)/RATE/0.01))
    v = np.floor(len(energy)/t)

    for x in range(1,t+1):
        mean.append(np.mean(energy[((x-1)*v):(x*v)-1]))
        eixo_tempo.append(np.ceil(v/2)+((x-1)*v))


    mean_db = map(lambda x: 10*np.log10(np.divide(x, maxenergy)), mean)
#     for i in range(0,len(mean)-1):
#         if mean[i]==0:
#             mean[i]= eps
#         mean_db.append(10*np.log10(np.divide(mean[i], maxenergy)))


    r=0
    for i in range(0, len(mean_db)):
        if (mean_db[i]>rms_dB+10) and (r<i):
            find.append(mean_db[i])
            r=i



    find=None
    find=array('f')
    for i in range(0,r):
        if mean_db[i]<rms_dB+10:
            find.append(i)

    if not find:
        r=0
    else:
        r=int(min(find))


    if r<10:
        r=10
    xi = eixo_tempo[0:r-1]
    A = np.vstack([xi, np.ones(len(xi))]).T
    y = mean_db[0:r-1]
    b, m = np.linalg.lstsq(A,y)[0]

    cruzamento = (rms_dB-m)/b
    ####################################################################
    if rms_dB > -20:
        #Relacao sinal ruido insuficiente
        ponto=len(energy);

    else:

        #%%%%%%%%%%%%%%%%%%%%%%%%    PARTE ITERATIVA DEL PROCESO    %%%%%%%

        erro=1.0;
        INTMAX=50.0;
        vezes=1.0;
        while ((erro > 0.0001) and (vezes <= INTMAX)):


            #Calculo de nuevos intervalos,
            #con p pasos por cada 10dB
            r=t=v=n=mean=eixo_tempo= None
            mean=array('f')
            eixo_tempo=array('f');          ' %numero de passos por decada'
            p = 5;
            if b==0:
                b=eps

            delta = abs(10/b);
            v = np.floor(delta/p);        '%intervalo para obtencao de mean'

            if cruzamento == np.inf:
                cruzamento = len(energy)

            t = int(np.floor(len(energy[0:round(cruzamento-delta)-1])/v))
            if t < 2:
                t=2;
            elif not t:
                t=2;


            for n in range(1,t+1):
                mean.append(np.mean(energy[(((n-1)*v)):(n*v)-1]))
                eixo_tempo.append(np.ceil(v/2)+((n-1)*v));

            meandB = [x*10 for x in np.log10(np.divide(mean, max(energy)))]

            xi=m=b=noise=rms_dB= None

            xi = eixo_tempo
            A = np.vstack([xi, np.ones(len(xi))]).T
            y=meandB
            b, m = np.linalg.lstsq(A,y)[0]


            noise = energy[round(cruzamento+delta):len(energy)]

            if (len(noise) < round(.1*len(energy))):
                noise = energy[round(.9*len(energy)):len(energy)-1]


            rms_dB = 10*np.log10(np.divide(np.mean(noise), max(energy)))
            erro = abs(cruzamento - (rms_dB-m)/b)/cruzamento;
            cruzamento = round((rms_dB-m)/b);
            vezes = vezes + 1;



    if cruzamento > len(energy):
        ponto = len(energy)
    else:
        ponto = cruzamento
    return ponto

def IR_window(ir):
    ir2=ir[ir.argmax():len(ir)-1]
    corte=lundeby(ir2)
    if corte>=1:
        irWindow=ir2[0:corte-1] #Escalar por 32768  y convertir a entero
    else:
        irWindow=ir2
    return np.int16(irWindow)


########## PROGRAMA #####################
 
sweep = SineSweep().generate()
figure()
plot(sweep)
show()
sweep_response = IRMeasurement(sweep).measure()

# ir = SineSweep.get_ir(sweep_response, sweep)

# impulse_response=IR_window(data)
# save_file(ir, RUTA+nombre_IR)
