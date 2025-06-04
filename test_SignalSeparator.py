from model_complex import SignalSeparator
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from load_dataset import SignalDataset
import commpy
import matplotlib.pyplot as plt
import pandas as pd

test_mode = 'qpsk_all_params_mode1'
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
model = SignalSeparator().to(device)
# model.load_state_dict(torch.load('signal_separator_'+test_mode+'.pth',weights_only=True))  # 加载模型权重
model.load_state_dict(torch.load('signal_separator_qpsk_all_params_200000.pth',weights_only=True))  # 加载模型权重
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
loaded_data = torch.load('/nas/datasets/yixin/PCMA/py_dataset/dataset_'+test_mode+'_test.pt')
# loaded_data = torch.load('/nas/datasets/yixin/PCMA/py_dataset/dataset_random_.pt')
dataset = SignalDataset(loaded_data)
batch_size = 32
test_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
fs = 93300
M = 4 
running_test_loss = 0.0
rolloff = 0
span = 2
delta_tao = 0.25


def up_sampling(waveform, zero_nums=1):
    upsampled_waveform = []
    for elem in waveform:
        upsampled_waveform.append(elem)
        for i in range(zero_nums):
            upsampled_waveform.append(0+0j)
    return upsampled_waveform
def rc_tx(waveform,rolloff,samples_per_symbol,span):
    coeffs = rcosdesign(rolloff,span,samples_per_symbol)
    upsampled_signal = up_sampling(waveform,samples_per_symbol-1)
    filtered_signal = np.convolve(upsampled_signal,coeffs,'full')
    delay = int((len(coeffs)-1)/2)
    filtered_signal = filtered_signal[:-delay*2]
    return filtered_signal
def rc_rx(waveform,rolloff,samples_per_symbol,span):
    coeffs = rcosdesign(rolloff,span,samples_per_symbol)
    delay = int((len(coeffs) - 1) / 2)
    filteredsignal = np.convolve(waveform, coeffs, 'full')
    filteredsignal_trimmed = filteredsignal[delay - samples_per_symbol:len(filteredsignal) - samples_per_symbol*2:samples_per_symbol]
    return filteredsignal_trimmed
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
def demodulate(signal, carrier_freq, oversampling_ratio,bits,random_phase_diff=0,random_delay=0,modulation_type="QPSK"):
    # n = int(oversampling_ratio * delta_tao)
    signal_phase_recovered = signal * np.exp(- 1j * random_phase_diff)
    n = random_delay
    signal_trimmed = np.zeros(len(signal_phase_recovered),dtype=signal_phase_recovered.dtype)
    if n !=0:
        signal_trimmed1 = signal_phase_recovered[n:]
        last_n = int(np.log2(M) * oversampling_ratio) - n
        signal_trimmed = signal_trimmed1[:-last_n]
        zeros = np.zeros(int(oversampling_ratio*2))
        signal_trimmed = np.append(signal_trimmed,zeros)
    else:
        signal_trimmed = signal_phase_recovered
    t = np.array(range(len(signal_trimmed))) /  (fs * oversampling_ratio)
    carrier = np.exp(2j * np.pi * carrier_freq * t)
    complex_baseband = signal_trimmed * carrier
    zeros_to_append = np.zeros(oversampling_ratio, dtype=complex_baseband.dtype)
    padded_complex_baseband = np.append(complex_baseband, zeros_to_append) / 0.8

    filteredsignal = rc_rx(padded_complex_baseband, rolloff, oversampling_ratio, span)
    filteredsignal_downsampled = filteredsignal[span::2]
    waveform = 0
    if modulation_type == "QPSK":
        qpsk = commpy.PSKModem(4)
        waveform = qpsk.modulate(bits)
    elif modulation_type =="8PSK":
        psk8 = commpy.PSKModem(8)
        waveform = psk8.modulate(bits)
    elif modulation_type == "16QAM":
        qam16 = commpy.QAMModem(16)
        waveform = qam16.modulate(bits)
    upsampled_waveform = up_sampling(waveform)
    waveform_rc = rc_tx(upsampled_waveform,rolloff,oversampling_ratio,span)
    bits_hat = qpsk.demodulate(filteredsignal_downsampled,demod_type='hard')
    bits_hat_trimmed = 0
    if delta_tao == 0.25:
        bits_trimmed = bits[:-int(np.log2(M))]
        bits_hat_trimmed = bits_hat[:-int(np.log2(M))]
    elif delta_tao == 0:
        bits_trimmed = bits
        bits_hat_trimmed = bits_hat
    ber = np.sum(bits_hat_trimmed != bits_trimmed)
    return ber
columns = ['snr', 'freq_overlap_percentage', 'amplititude_ratio', 'oversampling_ratio','random_phase_diff','random_delay']
result_df = pd.DataFrame(columns=columns)
bit_len = 128
is_first = 1
with torch.no_grad():
    for batch in test_loader:
        start = time.perf_counter()
        mixsignal_real = batch['mixsignal_real'].to(device).unsqueeze(1)
        mixsignal_imag = batch['mixsignal_imag'].to(device).unsqueeze(1)
        rfsignal1_real = batch['rfsignal1_real'].to(device).unsqueeze(1)
        rfsignal1_imag = batch['rfsignal1_imag'].to(device).unsqueeze(1)
        rfsignal2_real = batch['rfsignal2_real'].to(device).unsqueeze(1)
        rfsignal2_imag = batch['rfsignal2_imag'].to(device).unsqueeze(1)
        bits1 = batch['bits1']
        bits2 = batch['bits2']
        # (snr,freq_overlap_percetage,amplititude_ratio,oversampling_ratio,random_phase_diff,random_delay) = batch['params']
        (snr,freq_overlap_percetage,amplititude_ratio,oversampling_ratio,random_phase_diff,random_delay,modulation_type) = batch['params']
        origin_len = batch['origin_len']
        mixsignal = torch.cat([mixsignal_real,mixsignal_imag],dim=1)
        rfsignal1 = torch.cat([rfsignal1_real,rfsignal1_imag],dim=1)
        rfsignal2 = torch.cat([rfsignal2_real,rfsignal2_imag],dim=1)
        
        test_output = model(mixsignal)
        pred_rfsignal1 = torch.cat([test_output[0],test_output[1]],dim=1)
        pred_rfsignal2 = torch.cat([test_output[2],test_output[3]],dim=1)

        test_loss = torch.nn.MSELoss()(pred_rfsignal1, rfsignal1)/torch.sqrt(torch.mean(rfsignal1**2,dim=[1,2],keepdim=True)) + torch.nn.MSELoss()(pred_rfsignal2, rfsignal2)/torch.sqrt(torch.mean(rfsignal2**2,dim=[1,2],keepdim=True))
        test_loss_mean = test_loss.mean()
        running_test_loss += test_loss_mean.item() 

        for i in range(rfsignal1_real.size()[0]):
            n = origin_len[i].item()

            rfsignal1_real_i = rfsignal1_real[i].cpu().numpy()[0][:n]
            rfsignal1_imag_i = rfsignal1_imag[i].cpu().numpy()[0][:n]
            rfsignal1_i = rfsignal1_real_i + 1j * rfsignal1_imag_i
            rfsignal2_real_i = rfsignal2_real[i].cpu().numpy()[0][:n]
            rfsignal2_imag_i = rfsignal2_imag[i].cpu().numpy()[0][:n]
            rfsignal2_i = rfsignal2_real_i + 1j * rfsignal2_imag_i

            pred_rfsignal1_real_i = test_output[0][i].cpu().numpy()[0][:n]
            pred_rfsignal1_imag_i = test_output[1][i].cpu().numpy()[0][:n]
            pred_rfsignal1_i = pred_rfsignal1_real_i + 1j * pred_rfsignal1_imag_i
            pred_rfsignal2_real_i = test_output[2][i].cpu().numpy()[0][:n]
            pred_rfsignal2_imag_i = test_output[3][i].cpu().numpy()[0][:n]
            pred_rfsignal2_i = pred_rfsignal2_real_i + 1j * pred_rfsignal2_imag_i

            snr_i = snr[i].item()
            freq_overlap_percetage_i = freq_overlap_percetage[i].item()
            amplititude_ratio_i = float(str(amplititude_ratio[i].item())[:5])

            oversampling_ratio_i = oversampling_ratio[i].item()
            random_phase_diff_i = float(str(random_phase_diff[i].item())[:5])
            random_delay_i = float(str(random_delay[i].item())[:5])
            modulation_type_i = modulation_type[i].item()

            bits1_i = bits1[i].cpu().numpy()
            bits2_i = bits2[i].cpu().numpy()

            if is_first == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(pred_rfsignal1_real_i,label='pred sig1')
                plt.plot(rfsignal1_real_i, label='origin sig1')
                plt.title('Real Part')                
                plt.xlabel('SNR(dB)')
                plt.ylabel('BER')
                plt.grid(True)
                plt.legend()
                plt.savefig('./src/py_dataset/'+test_mode+'_real.png')
                plt.figure(figsize=(10, 6))
                plt.plot(pred_rfsignal1_imag_i,label='pred sig1')
                plt.plot(rfsignal1_imag_i, label='origin sig1')
                plt.title('Imag Part')                
                plt.xlabel('SNR(dB)')
                plt.ylabel('BER')
                plt.grid(True)
                plt.legend()
                plt.savefig('./src/py_dataset/'+test_mode+'_imag.png')
                is_first = 0
            carrier_freq1 = fs / 4
            B_rf = (1 + rolloff) * fs / oversampling_ratio_i
            delta_f = B_rf * (1 - freq_overlap_percetage_i / 100)
            carrier_freq2 = carrier_freq1 + delta_f

            ber1_i = demodulate(pred_rfsignal1_i, carrier_freq1, oversampling_ratio_i, bits1_i,modulation_type=modulation_type_i) / bit_len
            # ber2_i = demodulate(pred_rfsignal2_i, carrier_freq2, oversampling_ratio_i, bits2_i, random_delay=2) / bit_len
            ber2_i = demodulate(pred_rfsignal2_i, carrier_freq2, oversampling_ratio_i, bits2_i,random_phase_diff=random_phase_diff_i,random_delay=int(random_delay_i*2*oversampling_ratio_i),modulation_type=modulation_type_i) / bit_len
            ber1_i_ideal = demodulate(rfsignal1_i, carrier_freq1, oversampling_ratio_i, bits1_i,modulation_type=modulation_type_i) / bit_len
            ber2_i_ideal = demodulate(rfsignal2_i, carrier_freq2, oversampling_ratio_i, bits2_i,random_phase_diff=random_phase_diff_i, random_delay=int(random_delay_i*2*oversampling_ratio_i),modulation_type=modulation_type_i) / bit_len
            if ber1_i_ideal > 0 or ber2_i_ideal>0:
                raise('BER demodulate Error')

            ber = (ber1_i + ber2_i)/2
            new_row = {
                'snr':snr_i,
                'freq_overlap_percentage':freq_overlap_percetage_i,
                'amplititude_ratio':amplititude_ratio_i,
                'oversampling_ratio':oversampling_ratio_i,
                'random_phase_diff':random_phase_diff_i,
                'random_delay':random_delay_i,
                'BER':ber
            }
            new_row_df = pd.DataFrame([new_row])

            # 如果result_df是空的，需要先初始化
            if result_df.empty:
                result_df = new_row_df
            else:
                # 使用concat进行拼接
                result_df = pd.concat([result_df, new_row_df], ignore_index=True)
if test_mode == "qpsk_all_params_mode1":
    fixed_amplititude_ratio = 0.8
    fixed_oversampling_ratio = 8
    fixed_phase_diff = 0
    fixed_delay = 0.25


    filtered_df = result_df[
        (result_df['amplititude_ratio'] == fixed_amplititude_ratio) &
        (result_df['oversampling_ratio'] == fixed_oversampling_ratio) &
        (result_df['random_phase_diff'] == fixed_phase_diff) &
        (result_df['random_delay'] == fixed_delay)
    ]
    grouped_ber = filtered_df.groupby(['freq_overlap_percentage', 'snr'])['BER'].mean().reset_index()
    grouped = grouped_ber.groupby('freq_overlap_percentage')

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group['snr'], group['BER'], label=f'Freq Overlap {name}%')

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR-BER Curves for Different Freq Overlap Percentages')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./src/py_dataset/'+test_mode+'.png')
elif test_mode == "qpsk_all_params_mode2":
    fixed_amplititude_ratio = 0.8
    fixed_freq_overlap_percentage = 100
    fixed_phase_diff = 0
    fixed_delay = 0.25


    filtered_df = result_df[
        (result_df['amplititude_ratio'] == fixed_amplititude_ratio) &
        (result_df['freq_overlap_percentage'] == fixed_freq_overlap_percentage) &
        (result_df['random_phase_diff'] == fixed_phase_diff) &
        (result_df['random_delay'] == fixed_delay)
    ]
    grouped_ber = filtered_df.groupby(['oversampling_ratio', 'snr'])['BER'].mean().reset_index()
    grouped = grouped_ber.groupby('oversampling_ratio')

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group['snr'], group['BER'], label=f'OverSampling Ratio {name}')

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR-BER Curves for Different OverSampling Ratio')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./src/py_dataset/'+test_mode+'.png')
elif test_mode == "qpsk_all_params_mode3":
    fixed_oversampling_ratio = 8
    fixed_freq_overlap_percentage = 100
    fixed_phase_diff = 0
    fixed_delay = 0.25


    filtered_df = result_df[
        (result_df['oversampling_ratio'] == fixed_oversampling_ratio) &
        (result_df['freq_overlap_percentage'] == fixed_freq_overlap_percentage) &
        (result_df['random_phase_diff'] == fixed_phase_diff) &
        (result_df['random_delay'] == fixed_delay)
    ]
    grouped_ber = filtered_df.groupby(['amplititude_ratio', 'snr'])['BER'].mean().reset_index()
    grouped = grouped_ber.groupby('amplititude_ratio')

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group['snr'], group['BER'], label=f'Amplititude Ratio {name}')

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR-BER Curves for Different Amplititude Ratio')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./src/py_dataset/'+test_mode+'.png')
elif test_mode == "qpsk_all_params_mode4":
    fixed_amplititude_ratio = 0.8
    fixed_oversampling_ratio = 8
    fixed_freq_overlap_percentage = 100
    fixed_delay = 0.25


    filtered_df = result_df[
        (result_df['oversampling_ratio'] == fixed_oversampling_ratio) &
        (result_df['freq_overlap_percentage'] == fixed_freq_overlap_percentage) &
        (result_df['amplititude_ratio'] == fixed_amplititude_ratio) &
        (result_df['random_delay'] == fixed_delay)
    ]
    grouped_ber = filtered_df.groupby(['random_phase_diff', 'snr'])['BER'].mean().reset_index()
    grouped = grouped_ber.groupby('random_phase_diff')

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group['snr'], group['BER'], label=f'Phase Diff {name}%')

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR-BER Curves for Different Phase Diff')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./src/py_dataset/'+test_mode+'.png')
elif test_mode == "qpsk_all_params_mode5":
    fixed_amplititude_ratio = 0.8
    fixed_oversampling_ratio = 8
    fixed_freq_overlap_percentage = 100
    fixed_phase_diff = 0


    filtered_df = result_df[
        (result_df['oversampling_ratio'] == fixed_oversampling_ratio) &
        (result_df['freq_overlap_percentage'] == fixed_freq_overlap_percentage) &
        (result_df['amplititude_ratio'] == fixed_amplititude_ratio) &
        (result_df['random_phase_diff'] == fixed_phase_diff)
    ]
    grouped_ber = filtered_df.groupby(['random_delay', 'snr'])['BER'].mean().reset_index()
    grouped = grouped_ber.groupby('random_delay')

    plt.figure(figsize=(10, 6))
    for name, group in grouped:
        plt.plot(group['snr'], group['BER'], label=f'Delay {name}%')

    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR-BER Curves for Different Delay')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./src/py_dataset/'+test_mode+'.png')