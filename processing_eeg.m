clear all

% Load the data
data = load('BMIS_EEG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');

% Access the first channel
eeg_signals = double(data.data(1, :));

% Sampling frequency
fs_eeg = 250;

% Time vector
t = (0:length(eeg_signals)-1) / fs_eeg;

% Subtract the mean from the signal
eeg_signals = eeg_signals - mean(eeg_signals);

%% Plot time domain - Original Signal
figure;
subplot(2, 1, 1);
plot(t, eeg_signals);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original EEG Signal - Time Domain');
grid on;

%% Preprocessing - Notch Filter at 60 Hz
wo = 60 / (fs_eeg / 2);  % Normalize the frequency
bw = wo / 35;            % Bandwidth of the notch filter
[b, a] = iirnotch(wo, bw);

% Apply the notch filter
eeg_notched = filtfilt(b, a, eeg_signals);

%% Preprocessing - Low-Pass Filter at 100 Hz 
FcutLPF = 100;
[b, a] = butter(4, FcutLPF / (fs_eeg / 2), 'low');

% Apply the low-pass filter
eeg_filtered = filtfilt(b, a, eeg_notched);

% Plot filtered signal in time domain
subplot(2, 1, 2);
plot(t, eeg_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
title('Filtered EEG Signal (Notch + Low-Pass) - Time Domain');
grid on;

%% Frequency domain analysis - Original Signal
S_orig = fft(eeg_signals);
NFFT = length(S_orig);

% Define frequency resolution and frequency vector
F = fs_eeg / NFFT;
f = F * (0:NFFT-1);

% Calculate power spectral density (PSD)
P_orig = abs(S_orig).^2 / NFFT; % Normalize by NFFT to get power per Hz

% Plot the PSD of original signal
figure;
subplot(2, 1, 1);
plot(f, P_orig);
axis([0 100 0 max(P_orig)])     % Define the visualization range axis([xmin xmax ymin ymax])
xlabel('Frequency (Hz)');
ylabel('Power (uV^2/Hz)');
title('Original EEG Signal - Frequency Domain');
grid on;

%% Frequency domain analysis - Filtered Signal
S_filt = fft(eeg_filtered);
NFFT = length(S_filt);

% Calculate power spectral density (PSD)
P_filt = abs(S_filt).^2 / NFFT; % Normalize by NFFT to get power per Hz

% Plot the PSD of filtered signal
subplot(2, 1, 2);
plot(f, P_filt);
axis([0 100 0 max(P_filt)])     % Define the visualization range axis([xmin xmax ymin ymax])
xlabel('Frequency (Hz)');
ylabel('Power (uV^2/Hz)');
title('Filtered EEG Signal (Notch + Low-Pass) - Frequency Domain');
grid on;








% %% CMC
% DUR = 4; % Duration of segment (s)
% [S_x, S_y, S_xy, Fs] = compute_power_spectrum(double(eeg_filtered), double(emg_signals), DUR, fs_eeg, fs_emg);
% 
% 
% % Plotting
% figure;
% 
% % Plot dello spettro di potenza EEG (S_x)
% subplot(3, 1, 1);
% plot((0:size(S_x,1)-1)*(fs_eeg/DUR), S_x(:,1,1)); % Assumiamo il primo segmento e il primo canale
% title('Spettro di Potenza EEG (S_x)');
% xlabel('Frequenza (Hz)');
% ylabel('Potenza');
% 
% % Plot dello spettro di potenza EMG (S_y)
% subplot(3, 1, 2);
% plot((0:size(S_y,1)-1)*(fs_emg/DUR), S_y(:,1,1)); % Assumiamo il primo segmento e il primo canale
% title('Spettro di Potenza EMG (S_y)');
% xlabel('Frequenza (Hz)');
% ylabel('Potenza');
% 
% % Plot dello spettro di potenza incrociato (S_xy)
% subplot(3, 1, 3);
% plot((0:size(S_xy,1)-1)*(Fs/DUR), S_xy(:,1,1)); % Assumiamo il primo segmento e il primo canale
% title('Spettro di Potenza Incrociato (S_xy)');
% xlabel('Frequenza (Hz)');
% ylabel('Potenza');