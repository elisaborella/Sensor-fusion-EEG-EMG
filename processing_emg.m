clear all

% Directory containing your EMG data files
data_dir = 'BMIS_EMG_DATA\data\mat_data\';  % Adjust this path as needed

% External directory to save filtered data
filtered_data_dir = 'filtered_EMG_data\';  % Adjust this path as needed

% Create the external directory if it doesn't exist
if ~exist(filtered_data_dir, 'dir')
    mkdir(filtered_data_dir);
end

% List all .mat files in the directory and subdirectories
file_list = dir(fullfile(data_dir, '**', '*.mat'));

% Sampling frequency
fs_emg = 200;

% Notch filter parameters
wo = 60 / (fs_emg / 2);  % Normalize the frequency
bw = 0.8;            % Bandwidth of the notch filter
[b_notch, a_notch] = iirnotch(wo, bw);

% Band-pass filter parameters
Fcut1BPF = 20;
Fcut2BPF = 90; % Adjust this as per your requirements
Wn = [Fcut1BPF, Fcut2BPF] / (fs_emg / 2);  % Normalize cutoff frequencies
[b_bpf, a_bpf] = butter(4, Wn, 'bandpass');

% Iterate through each file
for file_idx = 1:numel(file_list)
    % Load the data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    data = load(file_path);
    
    % Get the variable name from the .mat file
    var_name = fieldnames(data);
    emg_signals = double(data.(var_name{1}));
    
    % Subtract the mean from each channel
    emg_signals = emg_signals - mean(emg_signals, 1);

    % Apply the notch filter to each channel
    emg_notched = filtfilt(b_notch, a_notch, emg_signals);

    % Apply the band-pass filter to each channel
    emg_filtered = filtfilt(b_bpf, a_bpf, emg_notched);

    % Determine where to save the filtered signals
    [~, relative_path] = fileparts(file_path);
    save_dir = fullfile(filtered_data_dir, relative_path);
    
    % Create the directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Save the filtered signals to a new .mat file
    save_path = fullfile(save_dir, [file_name(1:end-4) '_filtered.mat']);
    save(save_path, 'emg_filtered', 'fs_emg');
    
    % Print debug information
    fprintf('Saved filtered signals to: %s\n', save_path);
end



