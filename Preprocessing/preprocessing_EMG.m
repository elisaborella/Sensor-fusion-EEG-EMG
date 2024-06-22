function preprocessing_EMG
    %PREPROCESSING_EMG Preprocesses EMG data by filtering and normalizing
    %   This function reads raw EMG data from .mat files, applies band-pass
    %   and high-pass filters, normalizes the data, and saves the processed
    %   data to a specified directory.

    % Directory containing your EMG data files
    data_dir = 'Data\BMIS_EMG_DATA\data\mat_data\';  % Adjust this path as needed
    
    % Directory to save filtered data
    filtered_data_dir = 'Data\filtered_EMG_data\';  % Adjust this path as needed
    
    % Create the directory if it doesn't exist
    if ~exist(filtered_data_dir, 'dir')
        mkdir(filtered_data_dir);
    end
    
    % List all .mat files in the directory and subdirectories
    file_list = dir(fullfile(data_dir, '**', '*.mat'));
    
    % Sampling frequency for EMG data
    fs_emg = 200;
    
    % Band-pass filter parameters
    F1HPF = 10;  % Low cutoff frequency
    Fcut2BPF = 99;  % High cutoff frequency
    Wn = [F1HPF, Fcut2BPF] / (fs_emg / 2);  % Normalize cutoff frequencies
    [b_bpf, a_bpf] = butter(5, Wn, 'bandpass');  % Design the band-pass filter
    
    % High-pass filter parameters
    F1HPF = 2.5;  % Cutoff frequency for high-pass filter in Hz
    Wn = F1HPF / (fs_emg / 2);  % Normalize cutoff frequency
    [b_hpf, a_hpf] = butter(5, Wn, 'high');  % Design the high-pass filter
    
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
    
        % Keep the last 5 seconds of data
        emg_signals = emg_signals(size(emg_signals,1)-5*fs_emg:end,:);
        
        % Normalize the signals (z-score normalization)
        emg_signals = (emg_signals - mean(emg_signals, 1)) ./ std(emg_signals, 0, 1);
    
        % Ensure normalization did not introduce non-finite values
        emg_signals(~isfinite(emg_signals)) = 0;
        
        % Apply the band-pass filter to each channel
        emg_filtered = filtfilt(b_bpf, a_bpf, emg_signals);
    
        % Apply the high-pass filter to each channel
        emg_filtered = filtfilt(b_hpf, a_hpf, emg_filtered);
        
        % Save the filtered signal
        % Determine the save path for the segmented signals
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
end
