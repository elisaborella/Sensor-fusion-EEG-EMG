% Add the current folder and all its subfolders to Matlab path
addpath(genpath(pwd));

%% Filter EEG signal
preprocessing_EEG;

%% Filter EMG signal
preprocessing_EMG;

%% Plot filtered EEG signal
eeg_data = load("Data\filtered_EEG_data\S23_R1_G1\S23_R1_G1.mat");

plotter(eeg_data.eeg_filtered, eeg_data.fs_eeg, "EEG filtered signal");

clear eeg_data;
%% Plot filtered EMG signal
emg_data = load("Data\filtered_EMG_data\S23_R1_G1\S23_R1_G1.mat");

plotter(emg_data.emg_filtered, emg_data.fs_emg, "EMG filtered signal");

clear emg_data;
%% Plot Raw EEG signal
eeg_raw_data = load("Data\BMIS_EEG_DATA\data\mat_data\subject_23\S23_R1_G1.mat");

eeg_raw_signal = double(eeg_raw_data.data);
eeg_raw_signal = permute(eeg_raw_signal, [2 1]);
eeg_raw_signal = eeg_raw_signal - mean(eeg_raw_signal);

plotter(eeg_raw_signal, 250, "EEG raw signal");

clear eeg_raw_signal;

%% Plot Raw EMG signal
emg_raw_data = load("Data\BMIS_EEG_DATA\data\mat_data\subject_23\S23_R1_G1.mat");

emg_raw_signal = double(emg_raw_data.data);
emg_raw_signal = permute(emg_raw_signal, [2 1]);
emg_raw_signal = emg_raw_signal - mean(emg_raw_signal);

plotter(emg_raw_signal, 200, "EMG raw signal");

clear emg_raw_signal

%% Compute power spectrum
eeg_data = load("Data\filtered_EEG_data\S1_R1_G1\S1_R1_G1.mat");

emg_data = load("Data\filtered_EMG_data\S1_R1_G1\S1_R1_G1.mat");

[S_x, S_y, S_xy, fs] = compute_power_spectrum(eeg_data.eeg_filtered, emg_data.emg_filtered, eeg_data.fs_eeg, emg_data.fs_emg);

% Plot power spectrum
plot_power_spectrum(S_x, S_y, S_xy, eeg_data.fs_eeg);
clear [eeg_data, emg_data];
%% Compute CMC
CMC = compute_cmc(S_x, S_y, S_xy);

% plot CMC
plot_cmc(CMC, 1);

clear CMC;
%% EMG FEATURES EXTRACTION
extract_features('Data\filtered_EMG_data\','Data\EMG_features', 'emg');

%% EEG FEATURES EXTRACTION
extract_features('Data\filtered_EEG_data\','Data\EEG_features', 'eeg');

%% CMC features extraction
extract_cmc_feat('Data\filtered_EEG_data\', 'Data\filtered_EMG_data\', 'Data\CMC_features');