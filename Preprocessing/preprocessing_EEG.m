clear all

% Directory containing your EEG data files
data_dir = 'BMIS_EEG_DATA\data\mat_data\';  % Adjust this path as needed

% External directory to save filtered data
filtered_data_dir = 'unsegmented_filtered_EEG_data\';  % Adjust this path as needed

% Create the external directory if it doesn't exist
if ~exist(filtered_data_dir, 'dir')
    mkdir(filtered_data_dir);
end

% List all .mat files in the directory and subdirectories
file_list = dir(fullfile(data_dir, '**', '*.mat'));

% Sampling frequency
fs_eeg = 250;

% Notch filter parameters
wo = 60 / (fs_eeg / 2);  % Normalize the frequency
bw = 0.2;            % Bandwidth of the notch filter
[b_notch, a_notch] = iirnotch(wo, bw);

% Band-pass filter parameters
Fcut1BPF = 5;
Fcut2BPF = 50; % Adjust this as per your requirements
Wn = [Fcut1BPF, Fcut2BPF] / (fs_eeg / 2);  % Normalize cutoff frequencies
[b_bpf, a_bpf] = butter(5, Wn, 'bandpass');

% Segmentation parameters
window_length_ms = 550;
overlap_percentage = 60;
window_length_samples = round(window_length_ms / 1000 * fs_eeg);
stride_samples = round(window_length_samples * (1 - overlap_percentage / 100));

% Iterate through each file
for file_idx = 1:numel(file_list)
    % Load the data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    data = load(file_path);
    
    % Get the variable name from the .mat file
    var_name = fieldnames(data);
    eeg_signals = double(data.(var_name{1}));
    eeg_signals = permute(eeg_signals, [2 1]);

     % Remove or replace non-finite values
    eeg_signals(~isfinite(eeg_signals)) = 0; % Replace NaNs and Infs with 0
    
    % Normalize the signals to have mean 0 and standard deviation 1
    eeg_signals = (eeg_signals - mean(eeg_signals, 1)) ./ std(eeg_signals, 0, 1);

    % Apply the notch filter to each channel
    eeg_notched = filtfilt(b_notch, a_notch, eeg_signals);

    % Apply the band-pass filter to each channel
    eeg_filtered = filtfilt(b_bpf, a_bpf, eeg_notched);
    
    % Determine where to save the segmented signals
    [~, relative_path] = fileparts(file_path);
    save_dir = fullfile(filtered_data_dir, relative_path);
    
    % Create the directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    save_path = fullfile(save_dir, file_name);
    save(save_path, 'eeg_filtered', 'fs_eeg');
    
    % Print debug information
    fprintf('Saved filtered signal to: %s\n', save_path);
end