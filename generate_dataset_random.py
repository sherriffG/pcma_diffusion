import numpy as np
import random
import commpy
import torch
# import matplotlib.pyplot as plt

random.seed(425)
## 参数设置 
fs = 93300

modulation_type = 'QPSK'
# oversampling_ratios = [8]
# snr_range = list(range(0,11,1)) # 0 1 2 ... 10
# # snr_range = [10]
# amplititude_ratios = [0.8]
# freq_overlap_percentages = list(range(0,110,10))
bit_len = 128
phase_offset = 0
delta_tao = 0.25

## 成型滤波器参数
rolloff = 0
span = 2

def up_sampling(waveform, zero_nums=1):
    upsampled_waveform = []
    for elem in waveform:
        upsampled_waveform.append(elem)
        for i in range(zero_nums):
            upsampled_waveform.append(0+0j)
    return upsampled_waveform
def rcosdesign(rolloff,span,samples_per_symbol):
    t = np.linspace(-span/2, span/2, span*samples_per_symbol+1)
    coeffs = np.zeros_like(t)
    for idx in range(len(t)):
        t_val = t[idx]
        if np.isclose(t_val, 0.0):
            coeff = 1 - rolloff + (4 * rolloff) / np.pi
        else:
            theta_p = np.pi * t_val * (1 - rolloff)
            theta_m = np.pi * t_val * (1 + rolloff)
            
            numerator = np.sin(theta_p) + 4 * rolloff * t_val * np.cos(theta_m)
            denominator = np.pi * t_val * (1 - (4 * rolloff * t_val) **2 )
            
            if not np.isclose(denominator, 0.0, atol=1e-20):
                coeff = numerator / denominator
            else:
                coeff = rolloff / np.pi
        
        coeffs[idx] = coeff
    coeffs /= np.linalg.norm(coeffs)
    return coeffs
def rc_tx(waveform,rolloff,samples_per_symbol,span):
    coeffs = rcosdesign(rolloff,span,samples_per_symbol)
    upsampled_signal = up_sampling(waveform,samples_per_symbol-1)
    filtered_signal = np.convolve(upsampled_signal,coeffs,'full')
    delay = int((len(coeffs)-1)/2)
    filtered_signal = filtered_signal[:-delay*2]
    return filtered_signal

data_list = []
# numbers = np.logspace(np.log10(100), np.log10(100000), 10)
numbers = [200000]
for num_sample in numbers:
    num = int(num_sample)
    num_eight = 0
    for _ in range(num):
        ## mode1
        # oversampling_ratios = [8]
        # oversampling_ratio = random.choice(oversampling_ratios)
        # freq_overlap_percentage = random.randint(0,100)
        # amplititude_ratio = random.uniform(0.2,1)
        # snr = random.uniform(0,10)
        # random_phase_diff = 0
        # random_delay = 0


        ## all params 
        oversampling_ratios = range(4,25)
        oversampling_ratio = random.choice(oversampling_ratios) 
        if oversampling_ratio == 8:
            num_eight += 1
        freq_overlap_percentage = random.randint(0,100)
        amplititude_ratio = random.uniform(0.2,1)
        snr = random.uniform(0,10)
        random_phase_diff = np.random.uniform(0, np.pi/4)
        random_delay = np.random.randint(0, 2 * oversampling_ratio)

        # ## mode1_spec
        # oversampling_ratio = 8
        # freq_overlap_percentage = 80
        # amplititude_ratio = 0.8
        # snr = 10
        # random_phase_diff = 0
        # random_delay = 0

        B_rf = (1 + rolloff) * fs / oversampling_ratio
        bits1 = np.random.binomial(n=1,p=0.5,size=(bit_len))
        bits2 = np.random.binomial(n=1,p=0.5,size=(bit_len))
        waveform1 = 0
        waveform2 = 0
        if modulation_type == 'QPSK':
            qpsk = commpy.PSKModem(4)
            waveform1 = qpsk.modulate(bits1)
            waveform2 = qpsk.modulate(bits2)
        elif modulation_type == '8PSK':
            psk8 = commpy.PSKModem(8)
            waveform1 = psk8.modulate(bits1)
            waveform2 = psk8.modulate(bits2)
        elif modulation_type == '16QAM':
            qam16 = commpy.QAMModem(16)
            waveform1 = qam16.modulate(bits1)
            waveform2 = qam16.modulate(bits2)    
        waveform2 = waveform2 * np.exp(1j * random_phase_diff)
        upsampled_waveform1 = up_sampling(waveform1)
        upsampled_waveform2 = up_sampling(waveform2)

        waveform1_rc = rc_tx(upsampled_waveform1,rolloff,oversampling_ratio,span)
        waveform2_rc = rc_tx(upsampled_waveform2,rolloff,oversampling_ratio,span)

        # n = int(oversampling_ratio*delta_tao)
        n = random_delay
        carrier_freq1 = fs/4
        delta_f = B_rf * (1 - freq_overlap_percentage / 100)
        carrier_freq2 = carrier_freq1 + delta_f
        up_sampling_rate = fs * oversampling_ratio

        t = np.array(range(len(waveform1_rc))) / up_sampling_rate
        carrier1 = np.exp(-2j * np.pi * carrier_freq1 * t)
        carrier2 = np.exp(-2j * np.pi * carrier_freq2 * t) * amplititude_ratio
        max_len = int(128 / 2 * 2 * 24)
        
        rfsignal1 = waveform1_rc * carrier1
        rfsignal2 = waveform2_rc * carrier2
        if n!=0:
            rfsignal2_tao = np.concatenate((np.zeros(n),rfsignal2[:-n]))
        else:
            rfsignal2_tao = rfsignal2
        mixsignal = rfsignal1 + rfsignal2_tao
        padded_rfsignal1 = np.pad(rfsignal1, (0, max_len - len(rfsignal1)), mode='constant', constant_values=0)
        padded_rfsignal2_tao = np.pad(rfsignal2_tao, (0, max_len - len(rfsignal2)), mode='constant', constant_values=0)

        signal_power = np.mean(np.abs(mixsignal)**2)
        snr_linear = 10**(snr/10.0)

        noise_power = signal_power / snr_linear
        noise_stddev = np.sqrt(noise_power)
        noise = (noise_stddev/np.sqrt(2)) * (np.random.randn(*mixsignal.shape)+ 1j*np.random.randn(*mixsignal.shape))

        mixsignal_noisy = mixsignal + noise
        padded_mixsignal_noisy = np.pad(mixsignal_noisy,(0, max_len-len(mixsignal_noisy)), mode='constant',constant_values=0)
        params = (snr,freq_overlap_percentage,amplititude_ratio,oversampling_ratio,random_phase_diff,random_delay,modulation_type)
        origin_len = len(rfsignal1)
        new_entry = {
            'mixsignal':padded_mixsignal_noisy,
            # 'mixsignal_clean':mixsignal,
            'rfsignal1':padded_rfsignal1,
            'rfsignal2':padded_rfsignal2_tao,
            'params':params,
            'bits1':bits1,
            'bits2':bits2,
            'origin_len':origin_len,
        }
        data_list.append(new_entry)
    torch.save(data_list,'/nas/datasets/yixin/PCMA/py_dataset/dataset_qpsk_all_params_'+str(num)+'.pt')
    print('/nas/datasets/yixin/PCMA/py_dataset/dataset_qpsk_all_params_'+str(num)+'.pt saved')