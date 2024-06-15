clear all

% Directory containing your EEG data files
data_dir = 'BMIS_EEG_DATA\data\mat_data\';  % Adjust this path as needed

% External directory to save filtered data
filtered_data_dir = 'filtered_EEG_data\';  % Adjust this path as needed

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
bw = 0.8;            % Bandwidth of the notch filter
[b_notch, a_notch] = iirnotch(wo, bw);

% Low-pass filter parameters
FcutLPF = 100;
[b_lpf, a_lpf] = butter(4, FcutLPF / (fs_eeg / 2), 'low');

% Iterate through each file
for file_idx = 1:numel(file_list)
    % Load the data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    data = load(file_path);
    
    % Get the variable name from the .mat file
    var_name = fieldnames(data);
    eeg_signals = double(data.(var_name{1}));
    
    % Permute rows and columns of EEG signals
    eeg_signals = permute(eeg_signals, [2 1]);  % Swap rows and columns

    % Subtract the mean from each channel
    %eeg_signals = eeg_signals - mean(eeg_signals, 1);

    % Apply the notch filter to each channel
    eeg_notched = zeros(size(eeg_signals));
    for ch = 1:size(eeg_signals, 2)
        eeg_notched(:, ch) = filtfilt(b_notch, a_notch, eeg_signals(:, ch));
    end

    % Apply the low-pass filter to each channel
    eeg_filtered = zeros(size(eeg_signals));
    for ch = 1:size(eeg_signals, 2)
        eeg_filtered(:, ch) = filtfilt(b_lpf, a_lpf, eeg_notched(:, ch));
    end

    % Determine where to save the filtered signals
    [~, relative_path] = fileparts(file_path);
    save_dir = fullfile(filtered_data_dir, relative_path);
    
    % Create the directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Save the filtered signals to a new .mat file
    save_path = fullfile(save_dir, [file_name(1:end-4) '_filtered.mat']);
    save(save_path, 'eeg_filtered', 'fs_eeg');
    
    % Print debug information
    fprintf('Saved filtered signals to: %s\n', save_path);
end




