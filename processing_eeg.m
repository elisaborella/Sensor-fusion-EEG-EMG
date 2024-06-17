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
bw = 0.2;            % Bandwidth of the notch filter
[b_notch, a_notch] = iirnotch(wo, bw);

% Band-pass filter parameters
Fcut1BPF = 5;
Fcut2BPF = 50; % Adjust this as per your requirements
Wn = [Fcut1BPF, Fcut2BPF] / (fs_eeg / 2);  % Normalize cutoff frequencies
[b_bpf, a_bpf] = butter(5, Wn, 'bandpass');

% Segmentation parameters
window_length_ms = 550;
overlap_percentage = 10;
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


    % Subtract the mean from each channel
    %eeg_signals = eeg_signals - mean(eeg_signals, 1);

    % Apply the notch filter to each channel
    eeg_notched = filtfilt(b_notch, a_notch, eeg_signals);

    % Apply the band-pass filter to each channel
    eeg_filtered = filtfilt(b_bpf, a_bpf, eeg_notched);

    % Segment the filtered signals
    num_samples = size(eeg_filtered, 1);
    num_channels = size(eeg_filtered, 2);
    segments = {};
    
    start_idx = 1;
    while start_idx + window_length_samples - 1 <= num_samples
        end_idx = start_idx + window_length_samples - 1;
        segment = eeg_filtered(start_idx:end_idx, :);
        segments{end + 1} = segment;
        start_idx = start_idx + stride_samples;
    end

    % Determine where to save the segmented signals
    [~, relative_path] = fileparts(file_path);
    save_dir = fullfile(filtered_data_dir, relative_path);
    
    % Create the directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Save each segment to a new .mat file
    for seg_idx = 1:numel(segments)
        save_path = fullfile(save_dir, [file_name(1:end-4) '_filtered_segment_' num2str(seg_idx) '.mat']);
        eeg_segment = segments{seg_idx};
        save(save_path, 'eeg_segment', 'fs_eeg');
        
        % Print debug information
        fprintf('Saved filtered segment %d to: %s\n', seg_idx, save_path);
    end
end



