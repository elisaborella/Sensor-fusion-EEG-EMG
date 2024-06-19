clear all

% %% TEST
emg_data = load("unsegmented_filtered_EMG_data\S23_R1_G1\S23_R1_G1.mat");
eeg_data = load("unsegmented_filtered_EEG_data\S1_R5_G1\S1_R5_G1.mat");

emg_signal = emg_data.emg_filtered;
fs_emg = emg_data.fs_emg;
eeg_signal = eeg_data.eeg_filtered;
fs_eeg = eeg_data.fs_eeg;

plotter(emg_signal, fs_emg, "EMG signal");
plotter(eeg_signal, fs_eeg, "EEG signal");
%%
dur = 2;
[S_x, S_y, S_xy, fs] = compute_power_spectrum(eeg_signal, emg_signal, dur, fs_eeg, fs_emg);
plot_power_spectrum(S_x, S_y, S_xy, fs_eeg, dur);

ch = 1;

S_x_avg = mean(S_x(:, ch, :), 3);
S_y_avg = mean(S_y(:, ch, :), 3);
S_xy_avg = mean(S_xy(:, ch, :), 3);

% Calcolare la CMC
CMC = (abs(S_xy_avg).^2) ./ (S_x_avg .* S_y_avg);

figure;
plot(CMC);
xlabel('Frequency (Hz)');
ylabel('CMC');
title(sprintf('Magnitude Square Coherence (CMC) - Channel %d', ch));
grid on;

%% EMG FEATURES EXTRACTION
% Define paths and constants
data_directory = 'unsegmented_filtered_EMG_data\'; % Path to the directory containing subject data
file_list = dir(fullfile(data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories
sampling_frequency_emg = 200; % Hz

% Create directory for saving features
features_directory = 'EMG_features';
if ~exist(features_directory, 'dir')
    mkdir(features_directory);
end

for file_idx = 1:numel(file_list)
    % Load the filtered EMG data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    emg_data = load(file_path);

    % Access the filtered EMG data
    emg_filtered = emg_data.emg_filtered;

    % Perform feature extraction
    [features, parameters] = universal_feature_extraction(emg_filtered, sampling_frequency_emg, 'emg');

    % Construct save filename
    [~, base_name, ~] = fileparts(file_name);
    save_filename = fullfile(features_directory, [base_name '_features.mat']);

    % Save the extracted features
    save(save_filename, 'features', 'parameters');

    % Print debug information
    fprintf('Extracted features and saved to: %s\n', save_filename);
end



%% EEG FEATURES EXTRACTION
% Define paths and constants
data_directory = 'unsegmented_filtered_EEG_data\'; % Path to the directory containing subject data
file_list = dir(fullfile(data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories
sampling_frequency_eeg = 250; % Hz

% Create directory for saving features
features_directory = 'EEG_features';
if ~exist(features_directory, 'dir')
    mkdir(features_directory);
end

% Iterate through each filtered file
for file_idx = 1:numel(file_list)
    % Load the filtered EEG data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    eeg_data = load(file_path);

    % Access the filtered EEG data
    eeg_filtered = eeg_data.eeg_filtered;

    % Perform feature extraction
    [features, parameters] = universal_feature_extraction(eeg_filtered, sampling_frequency_eeg, 'eeg');

    % Construct save filename
    [~, base_name, ~] = fileparts(file_name);
    save_filename = fullfile(features_directory, [base_name '_features.mat']);

    % Save the extracted features
    save(save_filename, 'features', 'parameters');

    % Print debug information
    fprintf('Extracted features and saved to: %s\n', save_filename);
end
 





% %% Plot EMG Signals
% 
% % Load the data
% data_filtered = load('filtered_EMG_data\S1_R1_G1\S1_R1_G1_filtered_segment_1.mat');
% data_raw = load('BMIS_EMG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');
% 
% % Access the EMG signals before and after filtering
% var_name_raw = fieldnames(data_raw);
% if isempty(var_name_raw)
%     error('No fields found in data_raw structure.');
% end
% emg_signals_raw = double(data_raw.(var_name_raw{1}));  % Assuming rows are channels
% 
% var_name_filtered = fieldnames(data_filtered);
% if isempty(var_name_filtered)
%     error('No fields found in data_filtered structure.');
% end
% emg_signals_filtered = double(data_filtered.(var_name_filtered{1}));  % Assuming rows are channels
% 
% % Ensure both signals have the same number of samples
% min_samples = min(size(emg_signals_raw, 1), size(emg_signals_filtered, 1));
% disp(min_samples)
% emg_signals_raw = emg_signals_raw(1:min_samples, :);
% emg_signals_filtered = emg_signals_filtered(1:min_samples, :);
% 
% % Sampling frequency
% fs_emg = 200;
% 
% % Time vector
% t = (0:min_samples-1) / fs_emg;
% 
% % Plot some sample EMG signals in the time domain
% figure;
% num_channels_to_plot = min(size(emg_signals_raw, 1), 8);
% for i = 1:num_channels_to_plot
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 1);
%     plot(t, emg_signals_raw(:,i));
%     xlabel('Time (s)');
%     ylabel('Amplitude');
%     title(['Raw EMG Signal Channel ', num2str(i)]);
%     grid on;
%     
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 2);
%     plot(t, emg_signals_filtered(:, i));
%     xlabel('Time (s)');
%     ylabel('Amplitude');
%     title(['Filtered EMG Signal Channel ', num2str(i)]);
%     grid on;
% end
% 
% % Frequency domain analysis
% n = min_samples;
% f = (0:n-1)*(fs_emg/n);
% f = f(1:floor(n/2));
% 
% % Plot the frequency domain signals
% figure;
% for i = 1:num_channels_to_plot
%     % Compute FFT for raw signals
%     fft_raw = fft(emg_signals_raw(:,i));
%     P2_raw = abs(fft_raw/n);
%     P1_raw = P2_raw(1:floor(n/2));
%     P1_raw(2:end-1) = 2*P1_raw(2:end-1);
% 
%     % Compute FFT for filtered signals
%     fft_filtered = fft(emg_signals_filtered( :,i));
%     P2_filtered = abs(fft_filtered/n);
%     P1_filtered = P2_filtered(1:floor(n/2));
%     P1_filtered(2:end-1) = 2*P1_filtered(2:end-1);
% 
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 1);
%     plot(f, P1_raw);
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
%     title(['Raw EMG Signal Channel ', num2str(i)]);
%     grid on;
% 
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 2);
%     plot(f, P1_filtered);
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
%     title(['Filtered EMG Signal Channel ', num2str(i)]);
%     grid on;
% end
% 
% 
% 
% %% Plot EEG Signals
% 
% % Load the data
% data_filtered = load('filtered_EEG_data\S1_R1_G1\S1_R1_G1_filtered_segment_1.mat');
% data_raw = load('BMIS_EEG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');
% 
% % Access the EEG signals before and after filtering
% var_name_raw = fieldnames(data_raw);
% if isempty(var_name_raw)
%     error('No fields found in data_raw structure.');
% end
% eeg_signals_raw = double(data_raw.(var_name_raw{1}));  % Assuming rows are channels
% 
% var_name_filtered = fieldnames(data_filtered);
% if isempty(var_name_filtered)
%     error('No fields found in data_filtered structure.');
% end
% eeg_signals_filtered = double(data_filtered.(var_name_filtered{1}));  % Assuming rows are channels
% 
% % Ensure both signals have the same number of samples
% min_samples = min(size(eeg_signals_raw, 1), size(eeg_signals_filtered, 1));
% disp(min_samples)
% eeg_signals_raw = eeg_signals_raw(1:min_samples, :);
% eeg_signals_filtered = eeg_signals_filtered(1:min_samples, :);
% 
% % Sampling frequency
% fs_eeg = 250;
% 
% % Time vector
% t = (0:min_samples-1) / fs_eeg;
% 
% % Plot some sample EEG signals in the time domain
% figure;
% num_channels_to_plot = min(size(eeg_signals_raw, 1), 8);
% for i = 1:num_channels_to_plot
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 1);
%     plot(t, eeg_signals_raw(:,i));
%     xlabel('Time (s)');
%     ylabel('Amplitude');
%     title(['Raw EEG Signal Channel ', num2str(i)]);
%     grid on;
%     
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 2);
%     plot(t, eeg_signals_filtered(:, i));
%     xlabel('Time (s)');
%     ylabel('Amplitude');
%     title(['Filtered EEG Signal Channel ', num2str(i)]);
%     grid on;
% end
% 
% % Frequency domain analysis
% n = min_samples;
% f = (0:n-1)*(fs_eeg/n);
% f = f(1:floor(n/2));
% 
% % Plot the frequency domain signals
% figure;
% for i = 1:num_channels_to_plot
%     % Compute FFT for raw signals
%     fft_raw = fft(eeg_signals_raw(:,i));
%     P2_raw = abs(fft_raw/n);
%     P1_raw = P2_raw(1:floor(n/2));
%     P1_raw(2:end-1) = 2*P1_raw(2:end-1);
% 
%     % Compute FFT for filtered signals
%     fft_filtered = fft(eeg_signals_filtered( :,i));
%     P2_filtered = abs(fft_filtered/n);
%     P1_filtered = P2_filtered(1:floor(n/2));
%     P1_filtered(2:end-1) = 2*P1_filtered(2:end-1);
% 
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 1);
%     plot(f, P1_raw);
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
%     title(['Raw EEG Signal Channel ', num2str(i)]);
%     grid on;
% 
%     subplot(num_channels_to_plot, 2, (i-1)*2 + 2);
%     plot(f, P1_filtered);
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
%     title(['Filtered EEG Signal Channel ', num2str(i)]);
%     grid on;
% end

function smoothed_signal = moving_average(signal, window_size)
    % MOVING_AVERAGE Applica una media mobile al segnale
    smoothed_signal = filter(ones(1, window_size)/window_size, 1, signal);
end