'''
@author Javier Nistal
'''
import numpy as np

from scipy.fftpack import fft, ifft
from scipy.signal import chirp
from abc import ABCMeta, abstractmethod

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
    def __init__(
            self, 
            type='sweep', 
            sample_rate=44100, 
            duration=5, 
            method='logarithmic'
        ):
        
        super(SineSweep, self).__init__(type, sample_rate, duration)
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
