import sys
from array import array
import numpy as np

RATE = 44100

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