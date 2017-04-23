###############################################################################
#                                                                             #
#    Este programa implementa un método para la medición de respuestas al     #
#    impulso                                                                  #
#    mediante barrido sinusoidal logarítmico.                                 #
#                                                                             #
#    Universida Politécnica de Madrid                                         #    
#    Alumno:    Javier Nistal Hurlé                                           #
#    Tutor:    Lino García Morales                                            #
#                                                                             #
###############################################################################

import pyaudio
import wave
from array import array
from struct import pack
import numpy as np
from scipy.fftpack import fft, ifft
from pylab import plot, figure, show, subplot, title


########## CONSTANTES ##############
nombre_IR=r'\ir_samu.wav';
RUTA=r'C:\Users\Javi\AppData\Roaming\REAPER\Data\impulse_response'
SWEEP=r'\barrido.wav'
CHUNK = 512
WIDTH = 2
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 15
INPUT_DEVICE=1  
OUTPUT_DEVICE=4
FORMAT = pyaudio.paInt16
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
FRAME_MAX_VALUE = 2 ** 15 - 1
############# FUNCIONES ###################

#    Función que genera un archivo WAV con la respuesta al impulso.
#    data: array que contiene los datos de audio
#    path: variable string con la ruta donde guardar la respuesta al impulso

def save_file(data, path):
  
    data = pack('<' + ('h' * len(data)), *data)
        
    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(WIDTH)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()

#Función que genera el barrido sinusoidal y graba simultáneamente la respuesta
# a través del micrófono

def sineSweep():
    data_sweep = array('h')
    data_rec = array('h')
    
    print("* recording")
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                input_device_index=INPUT_DEVICE,
                output_device_index=OUTPUT_DEVICE,
                frames_per_buffer=CHUNK)
    
    wf=wave.open(RUTA+SWEEP, 'r')#Devuelve un string
    dataOUT=wf.readframes(CHUNK)
    i=0
    while dataOUT != '':  
        i=i+1
        stream.write(dataOUT, CHUNK)
        data_chunk=stream.read(CHUNK)
        
        data_sweep.fromstring(dataOUT)
        dataOUT=wf.readframes(CHUNK)
        data_rec.fromstring(data_chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()
    return data_rec, data_sweep

#    Funxión que implementa el método Lundeby para el enventanado de respuestas 
#    al impulso
#    ir: array que contiene la respuesta al impulso sin acotar

def lundeby(ir):
    energia=np.power(ir,2)
    
    rms_dB=10*np.log10(np.mean(np.divide(
        energia[round(0.9*len(energia)):len(energia)],max(energia))))
    
    t=int(np.floor(len(energia)/RATE/0.01))
    v=np.floor(len(energia)/t)  
    media=array('f')
    eixo_tempo=array('f')
    
    for x in range(1,t+1):
        media.append(np.mean(energia[((x-1)*v):(x*v)-1]))
        eixo_tempo.append(np.ceil(v/2)+((x-1)*v))
      
    mediadB=array('f')
    maxenergia=max(energia)
    
    for i in range(0,len(media)-1):
        if media[i]==0:
            media[i]=0.000000000000000000000000000001
        mediadB.append(10*np.log10(np.divide(media[i], maxenergia)))

    find=array('f')

    r=0
    for i in range(0, len(mediadB)):    
        if (mediadB[i]>rms_dB+10) and (r<i):
            find.append(mediadB[i])
            r=i
              
  
  
    find=None
    find=array('f')
    for i in range(0,int(r)):
        if mediadB[i]<rms_dB+10:
            find.append(i)
                      
    if not find:
        r=0
    else:
        r=np.int16(min(find))
  
      
    if r<10:
        r=10
    xi = eixo_tempo[0:r-1]
    A = np.vstack([xi, np.ones(len(xi))]).T 
    y=mediadB[0:r-1]
    b, m = np.linalg.lstsq(A,y)[0]

    cruzamento = (rms_dB-m)/b
    ####################################################################
    if rms_dB > -20:
        #Relacao sinal ruido insuficiente
        ponto=len(energia);
  
    else:
      
        #%%%%%%%%%%%%%%%%%%%%%%%%    PARTE ITERATIVA DEL PROCESO    %%%%%%%
      
        erro=1.0;
        INTMAX=50.0;
        vezes=1.0;
        while ((erro > 0.0001) and (vezes <= INTMAX)):
              
          
            #Cálculo de nuevos intervalos, 
            #con p pasos por cada 10dB
            r=t=v=n=media=eixo_tempo= None
            media=array('f')
            eixo_tempo=array('f');          ' %numero de passos por decada'
            p = 5;
            if b==0:
                b=0.00001                   
              
            delta = abs(10/b);
            v = np.floor(delta/p);        '%intervalo para obtencao de media'
            t = np.int(np.floor(len(energia[0:round(cruzamento-delta)-1])/v))
            if t < 2: 
                t=2;
            elif not t:
                t=2;
              
          
            for n in range(1,t+1):
                media.append(np.mean(energia[(((n-1)*v)):(n*v)-1]))
                eixo_tempo.append(np.ceil(v/2)+((n-1)*v));
              
            mediadB = [x*10 for x in np.log10(np.divide(media, max(energia)))]
              
            xi=m=b=noise=rms_dB= None
              
            xi = eixo_tempo
            A = np.vstack([xi, np.ones(len(xi))]).T 
            y=mediadB
            b, m = np.linalg.lstsq(A,y)[0]

  
            noise = energia[round(cruzamento+delta):len(energia)]
              
            if (len(noise) < round(.1*len(energia))):
                noise = energia[round(.9*len(energia)):len(energia)-1] 
                  
              
            rms_dB = 10*np.log10(np.divide(np.mean(noise), max(energia)))
            erro = abs(cruzamento - (rms_dB-m)/b)/cruzamento;
            cruzamento = round((rms_dB-m)/b);
            vezes = vezes + 1;
      
      
  
    if cruzamento > len(energia):
        ponto = len(energia)
    else:              
        ponto = cruzamento
    return ponto

# Función que implementa el filtrado inverso para la deconvolución del barrido
# data_out:    array que contiene la respuesta del sistema al barrido
# data_in:    array que contiene el barrido sinusoidal utilizado en la medición
def deconvolution (data_out, data_in):

    y=fft(data_out,len(data_out))
    x=fft(data_in,len(data_out))
 
    data=ifft(np.divide(y,x)).real
    data=np.around([x*32768 for x in data])
    
    return data
#Función que enventana la respuesta al impulso
#    ir:    respuesta al impulso sin enventanar
def IR_window(ir):
    ir2=ir[ir.argmax():len(ir)-1]
    corte=lundeby(ir2)
    if corte>=1:
        irWindow=ir2[0:corte-1] #Escalar por 32768  y convertir a entero
    else:
        irWindow=ir2
    return np.int16(irWindow)
########## PROGRAMA #####################
data_rec, data_sig=sineSweep()

data= deconvolution(data_rec, data_sig)

impulse_response=IR_window(data)
save_file(impulse_response, RUTA+nombre_IR)
figure(1)
subplot(3,1,1)
plot(data_rec)
title('Respuesta a la señal de excitación')
subplot(3,1,2)
plot(data)
title('Respuesta al impulso deconvolucionada')
subplot(3,1,3)
plot(impulse_response)
title('Respuesta al impulso enventanada')
show()