'''
#                                                                             #
#    This program measures the acoustic impulse response of a room using      #
#    logarithmic sine sweep                                                   #
#                                                                             #
#    Technical School of Madrid (UPM)                                         #
#    Author:    Javier Nistal Hurle                                           #
#    Supervisor:    Dr. Lino Garcia Morales                                   #
#    University: Technical School of Madrid (UPM)                             #
'''

import pyaudio
import wave
from array import array
from struct import pack
import numpy as np
from pylab import plot, figure, show, subplot, title
import numpy as np

from signals import SineSweep
from ir_window import IRwindow

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


if __name__=='__main__':
    
 
    sweep = SineSweep().create()

    sweep_response = IRMeasurement().measure(sweep)
    ir = SineSweep.get_ir(sweep, sweep_response)
    ir = IRwindow.lundeby(ir[np.argmax(ir):])
    plot(ir)
    show()
    
    print