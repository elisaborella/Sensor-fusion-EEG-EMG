clear all

% Directory containing your EMG data files
data_dir = 'BMIS_EMG_DATA\data\mat_data\';  % Adjust this path as needed

% External directory to save filtered data
filtered_data_dir = 'unsegmented_filtered_EMG_data\';  % Adjust this path as needed

% Create the external directory if it doesn't exist
if ~exist(filtered_data_dir, 'dir')
    mkdir(filtered_data_dir);
end

% List all .mat files in the directory and subdirectories
file_list = dir(fullfile(data_dir, '**', '*.mat'));

% Sampling frequency
fs_emg = 200;

% Band-pass filter parameters
Fcut1BPF = 10;
Fcut2BPF = 99; % Adjust this as per your requirements
Wn = [Fcut1BPF, Fcut2BPF] / (fs_emg / 2);  % Normalize cutoff frequencies
[b_bpf, a_bpf] = butter(5, Wn, 'bandpass');

% Segmentation parameters
window_length_ms = 550;
overlap_percentage = 0;

% Iterate through each file
for file_idx = 1:numel(file_list)
    % Load the data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    data = load(file_path);
    
    % Get the variable name from the .mat file
    var_name = fieldnames(data);
    emg_signals = double(data.(var_name{1}));

    % Replace non-finite values with the mean of the finite values
    for channel_idx = 1:size(emg_signals, 2)
        channel_data = emg_signals(:, channel_idx);
        finite_idx = isfinite(channel_data);
        
        % If there are finite values, replace non-finite values with their mean
        if any(finite_idx)
            mean_value = mean(channel_data(finite_idx));
            channel_data(~finite_idx) = mean_value;
        else
            % If all values are non-finite, set them to zero to avoid errors
            channel_data(:) = 0;
        end
        
        emg_signals(:, channel_idx) = channel_data;
    end
    %% SEGMENTATION
    [emg_segments, time, stride_samples, window_length_samples] = segmentation(emg_signals, fs_emg, 550, 0);

    % Moving Average in order to smooth the signal
    window_size = 100;
    emg_segments_smoothed = cellfun(@(seg) moving_average(seg, window_size), emg_segments, 'UniformOutput', false);
    
    %% RECONSTRUCTION
    emg_signal_reconstructed = zeros(size(emg_signals,1), size(emg_signals,2)); % Preallocation
    emg_cell = zeros(size(emg_signals,1),1);
    for i = 1:length(emg_segments_smoothed)
        % Prendi il segmento filtrato corrente dalla cella
        emg_segment_smoothed = emg_segments_smoothed{i};

        for j = 1:length(emg_segment_smoothed)
            segment_start_idx = (j - 1) * stride_samples + 1;
            segment_end_idx = segment_start_idx + window_length_samples - 1;
            % Assegna il segmento filtrato alla posizione corretta in emg_cell
            emg_cell(segment_start_idx:segment_end_idx) = emg_segment_smoothed{j};
        end
        emg_signal_reconstructed(:, i) = abs(emg_cell-mean(emg_cell));
    end

    % Normalize the signals (z-score normalization)
    emg_signals = (emg_signal_reconstructed - mean(emg_signal_reconstructed, 1)) ./ std(emg_signal_reconstructed, 0, 1);

    % Ensure normalization did not introduce non-finite values
    emg_signals(~isfinite(emg_signals)) = 0;

    %Apply the band-pass filter to each channel
    emg_filtered = filtfilt(b_bpf, a_bpf, emg_signals);
    
    %% SAVE FILTERED SIGNAL
    % Determine where to save the segmented signals
    [~, relative_path] = fileparts(file_path);
    save_dir = fullfile(filtered_data_dir, relative_path);
    
    % Create the directory if it doesn't exist
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    save_path = fullfile(save_dir, file_name);
    save(save_path, 'emg_filtered', 'fs_emg');
    
    % Print debug information
    fprintf('Saved filtered segment to: %s\n', save_path);
end