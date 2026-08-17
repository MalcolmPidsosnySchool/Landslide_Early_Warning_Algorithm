# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:42:23 2026

@author: josit
"""

# run %matplotlib qt in the console first



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import resample
from datetime import datetime, timedelta
import pandas as pd
import time as tm
import obspy
from glob import glob
import sys


########################################################################################################################
def RMS_Threshold(LTR, Thresh, minimum, quiet, loud, sign):
    A = sign * (Thresh-minimum) / 2
    d = (Thresh+minimum) / 2
    width = quiet/50
    centre = loud# + quiet
    return((2/np.pi) * A * np.arctan(width*(LTR-centre)) + d)
    

def DataUpdate(buffer, new_data):
    k = len(new_data)
    buffer[:-k] = buffer[k:]
    buffer[-k:] = new_data
    return(buffer)


def Gaussian(x, s2, mu):
    return(np.exp(-(x-mu)**2 / (2*s2)))

def Spike(x, tau_d, tau_r):
    return((np.exp(-x / tau_d) - np.exp(-x / tau_r)))

def SyntheticTrain(x, A, mu):
    return(A * np.sqrt(x - mu))

def SignalGenerator(x, k, f0, s2, A):
    weights = Gaussian(k, s2, f0)
    phases = np.random.uniform(0, 2*np.pi, len(k)) # This is done to randomize the sin phase and prevent constructive interference
    return(A * np.sum(weights[:, None] * np.sin(2 * np.pi * k[:, None] * x + phases[:, None]), axis = 0))
    

def RMS(array):
    return(np.sqrt(np.mean(np.square(array))))

########################################################################################################################

do_random_event = False
do_infrasound = True
do_seismic = True
do_recording = True
do_plotting = False
filter = True
use_gem_data = False
use_italy_data = False
use_meager_data = False

np.set_printoptions(precision=3)


########################################################################################################################

file_num = 16#int(sys.argv[1])
sensor = 0
num_sensors = 2
sensor_thresh = 2

homedir = '/home/malcolmp/masters_project/'
homedir = 'C:/Users/josit/OneDrive - Simon Fraser University (1sfu)/Grad School Research/'
gemdir = f'{homedir}GEM_Files/'
matlabdir = f'{homedir}Matlab_Files/'
Italydir = f'{matlabdir}Belli_Italy_DF/'
meagerdir = f'{homedir}Meager_Data/'

    

offset = 1000
delta = 1/100 #1/100
filter_width_high = 1e-1
filter_width_low = 4
high_pass = 2
low_pass = 20

short_term_length = 10  # seconds
long_term_length = 180 # seconds
buffer_length = 1800
event_counter_thresh = 15


RMS_quiet = 10 # The average Long Term RMS during quiet period; found during stream characterization
RMS_loud = 30 # The average Long Term RMS during loud periods; found during stream characterization
RMS_event_amp = 1.5 # maximum desired STA/LTA value
RMS_min_amp = 1 # minimum desired STA/LTA value
RMS_end_thresh_fraction = 0.75

differential_iterations = event_counter_thresh
gradient_thresh = 25 # mPa/s

amplitude_event_thresh_max = 8
amplitude_event_thresh_min = 3
amplitude_end_fraction = 0.5

variance_limit = 25

set_sec_length = 33 # How long a second will be equivalent to. 60 is one minute, 3600 is one hour
set_interval = 1 #1 # second
event_flag = False



########################################################################################################################
# Seismic Parameters
filter_width_high = 1e-1
filter_width_low = 4
high_pass_seis = 1
low_pass_seis = 40

seis_thresh = 1.5

########################################################################################################################

plt.rcParams.update({'font.size': 12})
plt.rcParams['agg.path.chunksize'] = 1000
dpi = 200
infra_lim = 1000
infra_min = -1000
left = 0.15
right = 0.86

k = np.arange(0.1, 20.1, 0.1)
s2 = 30000
mu = 1500
f0 = 6
tau_d = 120
tau_r = 0.9 * tau_d # 0.6 for no spike, 0.2 for spike
A = 100



sec_length = 100
interval = 60




# These times are for Cedar Creek Data
#time1 = '2025-08-15T12:40:00'
#time2 = '2025-08-18T08:00:00'

#time1 = '2025-09-01T00:00:00'
#time2 = '2025-09-04T00:00:00'

# These times are for Juan's Squamish River Data



#time1 = '2021-06-25T08:00:00'
#time2 = '2021-07-03T08:00:00'

# These are times for the Italy Data
if use_italy_data == False:
    delta = 1/50


#time1 = '2017-05-29T14:00:00' # Illgraben uses CEST during May, 29th. I fucking hate daylight savings time

N = int((1 / delta) * buffer_length)
buffer = np.zeros((4 + do_seismic, N), dtype=float)



########################################################################################################################

if use_italy_data == True:
    year = 2024
    all_pressure = []
    for sen in range(num_sensors):
        sensor = f'{sen}'.zfill(2)
        savedir = f'{Italydir}July {year}/'
        
        # 2023 times: '2023-07-12T06:00:00' - '2023-07-12T09:00:00'
        # 2024 July times: '2024-07-19T14:00:00' - '2024-07-19T17:00:00', '2024-07-19T18:10:00' - '2024-07-19T21:00:00'
        # 2024 June times: '2024-06-21T16:00:00' - '2024-06-21T21:30:00'
        
        time1 = '2024-07-19T18:10:00'
        time2 = '2024-07-19T21:00:00'
        
        time1 = '2024-06-21T16:00:00'
        time2 = '2024-06-21T21:30:00'
        
        #data_temp = obspy.read(f'{Italydir}July {year}/*{sensor}.HDF.{year}.201')
        data_temp = obspy.read(f'{Italydir}June {year}/*{sensor}.HDF.{year}.173')
        #data_temp = obspy.read(f'{Italydir}July {year}/*{sensor}.HDF.{year}.193')
        
        ## combine traces so that each station has one trace
        data_temp.merge()
        t1 = obspy.UTCDateTime(time1)
        t2 = obspy.UTCDateTime(time2)
        data_temp.trim(t1, t2)
        
        print("Ok we got here")
        for trace in data_temp:
            print("-" * 50)
            print(f"Trace ID: {trace.id}")   
            starttime=pd.Timestamp(trace.stats.starttime.datetime,tz='UTC')
            print(f"  Start time: {starttime}")
            endtime=pd.Timestamp(trace.stats.endtime.datetime,tz='UTC')
            print(f"  End time: {endtime}")
            print("-" * 50)
        trace_data = [trace.data for trace in data_temp]
        if trace_data:  # Ensure the list is not empty
            data = np.vstack(trace_data)
        times = (pd.date_range(start=starttime - pd.Timedelta(seconds = buffer_length),\
                                              end=endtime,periods=len(data[0]) + buffer_length/delta).values)
        pressure_data = ((data - np.median(data)) / 2.4855e5 * 1000)[0]
        all_pressure.append(pressure_data)
    
    
    
    if do_seismic == True:
        #data_temp = obspy.read(f'{Italydir}July {year}/*HHZ.{year}.201')
        data_temp = obspy.read(f'{Italydir}June {year}/*HHZ.{year}.173')
        #data_temp = obspy.read(f'{Italydir}July {year}/*HHZ.{year}.193')
        
        ## combine traces so that each station has one trace
        data_temp.merge()
        data_temp.trim(t1, t2)
        
        print("Ok we got here")
        for trace in data_temp:
            print("-" * 50)
            print(f"Trace ID: {trace.id}")   
            starttime=pd.Timestamp(trace.stats.starttime.datetime,tz='UTC')
            print(f"  Start time: {starttime}")
            endtime=pd.Timestamp(trace.stats.endtime.datetime,tz='UTC')
            print(f"  End time: {endtime}")
            print("-" * 50)
        trace_data = [trace.data for trace in data_temp]
        if trace_data:  # Ensure the list is not empty
            data = np.vstack(trace_data)
        times_seis = (pd.date_range(start=starttime - pd.Timedelta(seconds = buffer_length),\
                                              end=endtime,periods=len(data[0]) + buffer_length/delta).values)
                                
        
        seismic_data = ((data - np.median(data))/2.4855e8 * 1e6)[0]




else:
    do_seismic = False
    savedir = f'{matlabdir}Belli_Illgraben_DF/'
    
    all_pressure = []
    for sen in range(num_sensors):
        files = glob(f'{matlabdir}Belli_Illgraben_DF/ILG_*')
        files.sort()
        file_name = files[file_num]
        #file_name = f'{matlabdir}Belli_Illgraben_DF/ILG_20190811.mat'
        data=loadmat(file_name)
        M = data['M'][sen]
        tt = data['tt'].flatten() # We subtract 2/24 to convert to UTC time by subtracting 2 hours
        day = tt[0].astype(int)
        frac = tt[0] - day
        time1 = datetime.fromordinal(day) + timedelta(days=frac) - timedelta(days=366)
        day = tt[-1].astype(int)
        frac = tt[-1] - day
        time2 = datetime.fromordinal(day) + timedelta(days=frac) - timedelta(days=366)
        
        starttime = pd.Timestamp(time1, tz='UTC')
        endtime = pd.Timestamp(time2, tz='UTC')
        times = (pd.date_range(start=starttime - pd.Timedelta(seconds = buffer_length),\
                                              end=endtime,periods=len(M) + buffer_length/delta).values)
        pressure_data = (M - np.median(M)) * 1000 # There is a 400mV/Pa scaling, but I have no clue what the given units are for the data.
        # I THINK the units are already in Pa. So multiplying by 1000 gives mPa
        all_pressure.append(pressure_data)



########################################################################################################################

#day_buffer = pressure_data[0:24*3600*100]
#pressure_data = pressure_data[24*3600*100:]
#times = times[24*3600*100:]

counter = 0
event_counter = 0
event_times = []
event_ends = []
#amplitudes = np.zeros(amplitude_mean_length)
differential_RMS = np.zeros((4,differential_iterations))
differential_times = np.zeros((4,differential_iterations))

# Let's make some fake data
fake_data = np.zeros(len(pressure_data))
x = np.linspace(0, mu + s2 // 100, 100*mu + s2)
    
    
# For data recording
event_record = np.array([])
time_record = np.array([], dtype=np.datetime64)
infra_record = np.array([])
if do_seismic == True:
    seis_record = np.array([])
else:
    seis_filtered = np.ones(len(buffer[0]))
    seis_thresh = 0.5
    


########################################################################################################################
time_plot = times[0:N]

if do_plotting == True:
    plt.ion()
    fig, (ax, ax2) = plt.subplots(figsize=(8, 5),dpi = dpi, nrows = 2)
    fig.subplots_adjust(left = left, right = right, bottom = 0.15, top = 0.93, hspace = 0.6)   
    line, = ax.plot(time_plot, buffer[0], lw=2)
    line2, = ax2.plot(np.linspace(low_pass, high_pass, len(buffer[0])), buffer[0], lw=2)
    
    ax.set_ylim(infra_min, infra_lim)
    ax.set_xlim(time_plot[offset], time_plot[-offset])
    ax.set_ylabel('Filtered Amplitudes\n(mPa)')
    ax2.set_ylim(0,4)
    ax2.set_xlim(high_pass, low_pass)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude (mPa)')
    ax.tick_params(axis='x', labelrotation=25)
    if do_seismic == True:
        ax_seis = ax.twinx()
        line_seis, = ax_seis.plot(time_plot, buffer[4], lw=1.5, color='red', linestyle = '--', alpha =0.5)
        ax_seis.set_ylabel('Filtered Velocity (um/s)')
        ax_seis.set_ylim(-1000, 1000)
        
time1 = tm.time()
time2 = time1 + 1
time_diff = np.round((time2-time1),3)

short_start = int(-(1/delta)) * (1 + short_term_length) - offset
long_start = int(-(1/delta)) * (1 + long_term_length) - offset

counter+=int(interval/delta)

quiet_RMS_vals = []
event_RMS_vals = []


while (counter + int(interval/delta)) < len(all_pressure[0]):
    
    if do_seismic == True and (counter + int(interval/delta)) >= len(seismic_data):
        break # Seems like seismic data length does not always match up with pressure data length for Italy
    if counter >= N:
        interval = set_interval
        
        
     
    
    time_plot = times[counter:counter+N] 
    # current time (or infrasound) index + (100 indeces / second * 60 seconds / minute * time_diff pseudo minutes)
    time1 = np.round(tm.time(),3)
    for i in range(num_sensors - 1, -1, -1):
        new_data = all_pressure[i][counter:counter + int(interval/delta)] + fake_data[counter:counter + int(interval/delta)]
    #print(time_plot[0], time_plot[-1])    
        buffer[i] = DataUpdate(buffer[i], new_data)
    if do_seismic == True:
        new_data_seis = seismic_data[counter:counter + int(interval/delta)] #+ fake_data[counter:counter + int(interval/delta)]
        buffer[4] = DataUpdate(buffer[4], new_data_seis)
    
    RMS_fractions = []
    gradients = []
    FFT_fractions = []
    variances = []
    shorts = []
    longs = []
    
    RMS_event_thresh = 0
    amplitude_event_thresh = 0
    
 
    if filter == True:
        for i in range(num_sensors - 1, -1, -1):
            data_temp_filtered = buffer[i]
                
            data_ft = np.fft.rfft(data_temp_filtered)
            times_ft = np.fft.rfftfreq(len(time_plot), delta)
            
            taper = np.ones_like(times_ft) # Tukey filter or some shit like that, think of -arctan(x)
            idx = (times_ft < high_pass) & (times_ft > high_pass - filter_width_high)
            taper[idx] = 0.5 * (1 + np.cos(np.pi * (times_ft[idx] - high_pass) / filter_width_high))
            taper[times_ft <= high_pass - filter_width_high] = 0
            
            data_ft *= taper
            
            taper = np.ones_like(times_ft) # Tukey filter or some shit like that
            idx = (times_ft > low_pass) & (times_ft < low_pass + filter_width_low)
            taper[idx] = 0.5 * (1 + np.cos(np.pi * (times_ft[idx] - low_pass) / filter_width_low))
            taper[times_ft >= low_pass + filter_width_low] = 0
            
            data_ft *= taper # We gotta do a low_pass as well due to the disconnects causing huge signals with "high frequencies"
            
            infra_filtered = np.fft.irfft(data_ft)
            
            A = np.abs(data_ft) / len(buffer[0])
            A[1:-1] *= 2
            
            # Do the same, but only the last {short_term_length} seconds and {long_term_length} seconds
            
            data_temp_filtered = infra_filtered[short_start:-offset]
                
            data_ft = np.fft.rfft(data_temp_filtered)
            times_ft_short = np.fft.rfftfreq(len(time_plot[short_start:-offset]), delta)
            
            A_short = np.abs(data_ft) / len(buffer[0][short_start:-offset])
            A_short[1:-1] *= 2
            
            
            data_temp_filtered = infra_filtered[long_start:-offset]
                
            data_ft = np.fft.rfft(data_temp_filtered)
            times_ft_long = np.fft.rfftfreq(len(time_plot[long_start:-offset]), delta)
            
            A_long = np.abs(data_ft) / len(buffer[0][long_start:-offset])
            A_long[1:-1] *= 2
            
            
            
            short_RMS = RMS(infra_filtered[short_start:-offset]) # This is the RMS of the last short_term_length seconds of infrasound    
            shorts.append(short_RMS)
            long_RMS = RMS(infra_filtered[long_start:-offset]) # This is the RMS of the last long_term_length seconds of infrasound
            longs.append(long_RMS)
            RMS_fractions.append(short_RMS/long_RMS)
            RMS_event_thresh += (RMS_Threshold(long_RMS, RMS_event_amp, RMS_min_amp, RMS_quiet, RMS_loud ,-1) - RMS_event_thresh) / (i+1)
              
            differential_RMS[i] = DataUpdate(differential_RMS[i], [short_RMS])
            differential_times[i] = DataUpdate(differential_times[i], [counter * delta])
            gradients.append(np.gradient(differential_RMS[i], differential_times[i]))
            
            short_frequency_mask = (times_ft_short >= 2) & (times_ft_short <= 8)
            long_frequency_mask = (times_ft_long >= 2) & (times_ft_long <= 8)
            all_frequency_mask = (times_ft >= 2) & (times_ft <= 8)
            short_fft_mean = np.nanmean(A_short[short_frequency_mask])
            long_fft_mean = np.nanmean(A_long[long_frequency_mask])
            FFT_fractions.append(short_fft_mean/long_fft_mean)
            #amplitudes = DataUpdate(amplitudes, [FFT_fraction])
            amplitude_event_thresh += (RMS_Threshold(long_RMS, amplitude_event_thresh_max, amplitude_event_thresh_min, RMS_quiet, RMS_loud ,-1) - amplitude_event_thresh) / (i+1)
            
            
            variance_mask = (A_long[long_frequency_mask] > 1) #np.median(A_short[short_frequency_mask]))
            if A_long[long_frequency_mask][variance_mask].size == 0:
                variances.append(0)
            else:
                variances.append((np.nanstd(A_long[long_frequency_mask][variance_mask]))**2)
        
        if do_seismic == True:
            data_temp_filtered = buffer[4]
                
            data_ft_seis = np.fft.rfft(data_temp_filtered)
            times_ft_seis = np.fft.rfftfreq(len(time_plot), delta)
            
            taper = np.ones_like(times_ft) # Tukey filter or some shit like that, think of -arctan(x)
            idx = (times_ft < high_pass) & (times_ft > high_pass_seis - filter_width_high)
            taper[idx] = 0.5 * (1 + np.cos(np.pi * (times_ft[idx] - high_pass_seis) / filter_width_high))
            taper[times_ft <= high_pass_seis - filter_width_high] = 0
            
            data_ft_seis *= taper
            
            taper = np.ones_like(times_ft) # Tukey filter or some shit like that
            idx = (times_ft > low_pass) & (times_ft < low_pass_seis + filter_width_low)
            taper[idx] = 0.5 * (1 + np.cos(np.pi * (times_ft[idx] - low_pass_seis) / filter_width_low))
            taper[times_ft >= low_pass_seis + filter_width_low] = 0
            
            data_ft_seis *= taper # We gotta do a low_pass as well due to the disconnects causing huge signals with "high frequencies"
            
            seis_filtered = np.fft.irfft(data_ft_seis)
            
            A_seis = np.abs(data_ft) / len(buffer[4])
            A_seis[1:-1] *= 2
            
            short_RMS_seis = RMS(seis_filtered[short_start:-offset])
            long_RMS_seis = RMS(seis_filtered[long_start:-offset])
            RMS_frac_seis = short_RMS_seis/long_RMS_seis
        else:
            RMS_frac_seis = 1
            
        
    else:
        infra_filtered = [buffer][0]
     
    if do_plotting == True:
        line.set_ydata(infra_filtered[offset:-offset])     # update plot data
        line.set_xdata(time_plot[offset:-offset])
        ax.set_xlim(time_plot[offset], time_plot[-offset])
        
        line2.set_ydata(A)
        line2.set_xdata(times_ft)
        ax2.set_xlim(high_pass, low_pass)
        
        if do_seismic == True:
            line_seis.set_ydata(seis_filtered[offset:-offset])     # update plot data
            line_seis.set_xdata(time_plot[offset:-offset])
        
        
        #plt.pause(0.5)
        fig.canvas.draw()
        fig.canvas.flush_events()
        

    
    
    # Shift to the left 1 second to avoid errors generated at the start/end of the data by fourier filtering 
    # Keep in mind this test does not show EVERY short term RMS due to each update being time_diff minutes of data
    # But the short term RMS is only the last few seconds of data
    

    
    amplitude_end_thresh = amplitude_end_fraction * amplitude_event_thresh
    

    print(f'Short Term RMS: {short_RMS:.3f}; Long Term RMS: {long_RMS:.3f}\
         \nFraction: {np.mean(RMS_fractions):.3f}; RMS Threshold: {RMS_event_thresh:.3f}\
         \nGradient: {np.mean(np.amax(np.abs(gradients), axis = 1)):.3f}mPa/s \n{"-"*60}\
         \nShort Term FFT Mean: {short_fft_mean:.3f}; Long Term FFT Mean: {long_fft_mean:.3f}\
         \nFraction: {np.mean(FFT_fractions):.3f}; Threshold: {amplitude_event_thresh:.3f}\
         \nVariance: {np.mean(variances):.3f}\n{"-"*60}\
         \nSeismic Mean = {np.mean(np.abs(seis_filtered[short_start: -offset])):.3f}\
         \nSeismic STA/LTA = {RMS_frac_seis:.3f}\n{"-"*60}\
         \nCurrent Time: {time_plot[-101]}\
         \nTime Taken: {time_diff} seconds\
         \nTime Passed: {(interval):.2f} seconds')   
         

    print(f'Long RMS Values: {np.array(longs)}')
    if (np.sum(np.array(variances) <= 0.1) == num_sensors)\
        and  (np.sum(np.amax(np.abs(gradients), axis = 1) <= 1) == num_sensors):
            quiet_RMS_vals.append(longs)
    if counter >= N:    
        if event_flag == False:
            print('No Event')
            if event_counter >= event_counter_thresh:
                event_flag = True
                event_counter = 0
                event_start_time = time_plot[-offset]
                event_times.append(event_start_time)
                
                #((RMS_fraction >= RMS_event_thresh\
                #or FFT_fraction >= amplitude_event_thresh)
            elif ((np.sum(FFT_fractions >= amplitude_event_thresh) >= sensor_thresh\
            and np.sum(RMS_fractions >= RMS_event_thresh)) >= sensor_thresh\
            and np.sum(np.array(variances) <= variance_limit) >= sensor_thresh\
            and np.sum(np.amax(np.abs(gradients), axis = 1) <= gradient_thresh) >= sensor_thresh\
            and np.mean(shorts) >= (RMS_loud / 2)\
            and RMS_frac_seis >= seis_thresh):
                event_counter += 1
            else:
                event_counter = 0
        else:
            print('\033[31mEvent\033[0m')
            event_RMS_vals.append(longs)
            if event_counter >= event_counter_thresh:
                event_flag = False
                event_counter = 0
                event_end_time = time_plot[-offset]
                event_ends.append(event_end_time)
            elif np.sum(FFT_fractions <= amplitude_end_thresh) >= sensor_thresh\
                or np.sum(np.array(variances) > variance_limit) >= sensor_thresh\
                or np.sum(RMS_fractions <= RMS_end_thresh_fraction * RMS_event_thresh) >= sensor_thresh:
                event_counter += 1
            else:
                event_counter = 0           
    print(f'{"#"*60}')
    time_record = np.append(time_record, time_plot[-len(new_data) - offset])
    event_record = np.append(event_record, event_flag)
    infra_record = np.append(infra_record, infra_filtered[-len(new_data) - offset])#np.mean(infra_filtered[-len(new_data) - offset:-offset])) # 
    if do_seismic == True:
        seis_record = np.append(seis_record, seis_filtered[-len(new_data_seis) - offset])#np.mean(seis_filtered[-len(new_data) - offset:-offset]))
    

    
    #tm.sleep(0.025)
    time1_extra=time1
    time2 = np.round(tm.time(),3)
    counter+=int(interval/delta)
    time_diff = np.round((time2-time1),3)
    
    
    
if do_recording == True:
    
    times_str = np.datetime_as_string(time_record.astype('datetime64[ms]'), unit='ms')
    if do_seismic == True:
        seis_event_data = np.empty(len(times_str), dtype=[('time','U30'),('velocity', 'f8'), ('events', 'bool')])
        seis_event_data['time'] = times_str
        seis_event_data['velocity'] = seis_record
        seis_event_data['events'] = event_record
    
    event_data = np.empty(len(times_str), dtype=[('time','U30'),('infrasound','f8'), ('events', 'bool')])
    event_data['time'] = times_str
    event_data['infrasound'] = infra_record
    event_data['events'] = event_record


    ########################################################################################################################
    t1 = starttime
    t2 = endtime
    if do_random_event == True:
        savefile = f'{savedir}Infrasound_All_event_{t1.year}_{t1.month}_{t1.day}-{t2.year}_{t2.month}_{t2.day}.csv'
        if do_seismic == True:
            seis_savefile = f'{savedir}Seismic_All_event_{t1.year}_{t1.month}_{t1.day}-{t2.year}_{t2.month}_{t2.day}.csv'
            np.savetxt(seis_savefile, seis_event_data, delimiter=',', header='datetimes, velocity, events', fmt=['%s','%.6f', '%s'])
    else:
        savefile = f'{savedir}Infrasound_All_{t1.year}_{t1.month}_{t1.day}-{t2.year}_{t2.month}_{t2.day}.csv'
        if do_seismic == True:
            seis_savefile = f'{savedir}Seismic_All_{t1.year}_{t1.month}_{t1.day}-{t2.year}_{t2.month}_{t2.day}.csv'
            np.savetxt(seis_savefile, seis_event_data, delimiter=',', header='datetimes, velocity, events', fmt=['%s','%.6f', '%s'])
            
    np.savetxt(savefile, event_data, delimiter=',', header='datetimes, infrasound, events', fmt=['%s','%.6f', '%s'])
    
 
print(f'Number of Events: {len(event_times)}\nEvent Times: {event_times}\nEvent Ends: {event_ends}')

event_RMS_vals = np.array(event_RMS_vals)
quiet_RMS_vals = np.array(quiet_RMS_vals)

for i in range(num_sensors):
    print(f'Quiet RMS: {np.mean(quiet_RMS_vals[:,i])}; Event RMS: {np.mean(event_RMS_vals[:,i])}; Signal-to-Noise: {np.mean(event_RMS_vals[:,i])/np.mean(quiet_RMS_vals[:,i])}')


