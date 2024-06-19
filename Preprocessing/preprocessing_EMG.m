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

% Notch filter parameters
wo = 60 / (fs_emg / 2);  % Normalize the frequency
bw = 0.2;            % Bandwidth of the notch filter
[b_notch, a_notch] = iirnotch(wo, bw);

% Segmentation parameters
window_length_ms = 400;
overlap_percentage = 50;

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
    
    first_window_size = 200;
    second_window_size = 20;
    emg_signals_smoothed = zeros(size(emg_signals,1), size(emg_signals,2));
    for i = 1:size(emg_signals,2)
        emg_signals_smoothed(:,i) = moving_average_array(emg_signals(:,i), first_window_size);
        emg_signals_smoothed(:,i) = moving_average_array(emg_signals_smoothed(:,i), second_window_size);
    end
    % Normalize the signals (z-score normalization)
    emg_signals = (emg_signals_smoothed - mean(emg_signals_smoothed, 1)) ./ std(emg_signals_smoothed, 0, 1);

    % Ensure normalization did not introduce non-finite values
    emg_signals(~isfinite(emg_signals)) = 0;

    % Apply the notch filter to each channel
    emg_notched = filtfilt(b_notch, a_notch, emg_signals);

    % Apply the band-pass filter to each channel
    emg_filtered = filtfilt(b_bpf, a_bpf, emg_notched);
    
%     rms_window_length = 50;
%     rms_window = dsp.movingRMS(rms_window_length);
% 
%     emg_filtered = rms_window(rms_window_length);
    
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

function rms_signal = rms_calc(signal, window_size)
    % RMS_CALC Calculate the RMS of the signal with a moving window
    rms_signal = sqrt(movmean(signal.^2, window_size));
end