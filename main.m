clear all

% % Define paths and constants
% data_directory = 'filtered_EMG_data\'; % Path to the directory containing subject data
% file_list = dir(fullfile(data_directory, '**', '*_filtered.mat')); % List all filtered .mat files in subdirectories
% sampling_frequency_emg = 200; % Hz
% 
% % Create directory for saving features
% features_directory = 'EMG_features';
% if ~exist(features_directory, 'dir')
%     mkdir(features_directory);
% end
% 
% % Iterate through each filtered file
% for file_idx = 1:numel(file_list)
%     % Load the filtered EMG data
%     file_name = file_list(file_idx).name;
%     file_path = fullfile(file_list(file_idx).folder, file_name);
%     emg_data = load(file_path);
% 
%     % Access the filtered EMG data
%     emg_filtered = emg_data.emg_filtered;
% 
%     % Perform feature extraction
%     [features, parameters] = universal_feature_extraction(emg_filtered, sampling_frequency_emg, 'emg');
% 
%     % Construct save filename
%     [~, base_name, ~] = fileparts(file_name);
%     save_filename = fullfile(features_directory, [base_name '_features.mat']);
% 
%     % Save the extracted features
%     save(save_filename, 'features', 'parameters');
% 
%     % Print debug information
%     fprintf('Extracted features and saved to: %s\n', save_filename);
% end
% 
% 
% % Define paths and constants
% data_directory = 'filtered_EEG_data\'; % Path to the directory containing subject data
% file_list = dir(fullfile(data_directory, '**', '*_filtered.mat')); % List all filtered .mat files in subdirectories
% sampling_frequency_eeg = 250; % Hz
% 
% % Create directory for saving features
% features_directory = 'EEG_features';
% if ~exist(features_directory, 'dir')
%     mkdir(features_directory);
% end
% 
% % Iterate through each filtered file
% for file_idx = 1:numel(file_list)
%     % Load the filtered EEG data
%     file_name = file_list(file_idx).name;
%     file_path = fullfile(file_list(file_idx).folder, file_name);
%     eeg_data = load(file_path);
% 
%     % Access the filtered EEG data
%     eeg_filtered = eeg_data.eeg_filtered;
% 
%     % Perform feature extraction
%     [features, parameters] = universal_feature_extraction(eeg_filtered, sampling_frequency_eeg, 'eeg');
% 
%     % Construct save filename
%     [~, base_name, ~] = fileparts(file_name);
%     save_filename = fullfile(features_directory, [base_name '_features.mat']);
% 
%     % Save the extracted features
%     save(save_filename, 'features', 'parameters');
% 
%     % Print debug information
%     fprintf('Extracted features and saved to: %s\n', save_filename);
% end
% % 





%% Plot EMG signals
% Load the data
data = load('filtered_EMG_data\S1_R1_G1\S1_R1_G1_filtered.mat');

% Access the EMG signals
emg_signals = data.emg_filtered;

% Sampling frequency
fs_emg = 200;

% Time vector
t = (0:size(emg_signals, 1)-1) / fs_emg;

% Plot some sample EMG signals
figure;
for i = 1:min(size(emg_signals, 2), 8) % Plot up to the first 4 channels
    subplot(min(size(emg_signals, 2), 8), 1, i);
    plot(t, emg_signals(:, i));
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['EMG Signal Channel ', num2str(i)]);
    grid on;
end


%% Plot EEG Signals

% Load the data
data = load('filtered_EEG_data\S1_R1_G1\S1_R1_G1_filtered.mat');

% Access the EEG signals
eeg_signals = data.eeg_filtered;

% Sampling frequency
fs_eeg = 250;

% Time vector
t = (0:size(eeg_signals, 1)-1) / fs_eeg;

% Plot some sample EEG signals
figure;
for i = 1:min(size(eeg_signals, 2), 8) % Plot up to the first 4 channels
    subplot(min(size(eeg_signals, 2), 8), 1, i);
    plot(t, eeg_signals(:, i));
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['EEG Signal Channel ', num2str(i)]);
    grid on;
end



%% Preprocessing

%% CMC
DUR = 4; % Duration of segment (s)
clear S_x;
clear S_y;
clear S_xy;
[S_x, S_y, S_xy] = compute_power_spectrum(eeg_signals, double(emg_signals), DUR, fs_eeg, fs_emg);

temp = S_x .* S_y;
norma = abs(S_xy)^2;
CMC =  norma / temp;