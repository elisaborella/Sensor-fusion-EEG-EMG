function preprocessing_EEG
    %PREPROCESSING_EEG Preprocesses EEG data by filtering and normalizing
    %   This function reads raw EEG data from .mat files, applies notch,
    %   band-pass, and low-pass filters, normalizes the data, removes
    %   padding, and saves the processed data to a specified directory.

    % Directory containing your EEG data files
    data_dir = 'Data\BMIS_EEG_DATA\data\mat_data\';  % Adjust this path as needed
    
    % Directory to save filtered data
    filtered_data_dir = 'Data\filtered_EEG_data\';  % Adjust this path as needed
    
    % Create the directory if it doesn't exist
    if ~exist(filtered_data_dir, 'dir')
        mkdir(filtered_data_dir);
    end
    
    % List all .mat files in the directory and subdirectories
    file_list = dir(fullfile(data_dir, '**', '*.mat'));
    
    % Sampling frequency for EEG data
    fs_eeg = 250;
    
    % Notch filter parameters
    wo = 60 / (fs_eeg / 2);  % Normalize the frequency
    bw = 0.2;  % Bandwidth of the notch filter
    [b_notch, a_notch] = iirnotch(wo, bw);
    
    % Band-pass filter parameters
    Fcut1BPF = 2;  % Low cutoff frequency
    Fcut2BPF = 50;  % High cutoff frequency
    Wn = [Fcut1BPF, Fcut2BPF] / (fs_eeg / 2);  % Normalize cutoff frequencies
    [b_bpf, a_bpf] = butter(5, Wn, 'bandpass');  % Design the band-pass filter
    
    % Low-pass filter parameters (to remove frequencies above 60 Hz)
    Fcut_lp = 60;  % Cutoff frequency for low-pass filter
    Wn_lp = Fcut_lp / (fs_eeg / 2);  % Normalize cutoff frequency for low-pass
    [b_lp, a_lp] = butter(5, Wn_lp, 'low');  % Design the low-pass filter
    
    % Padding parameters (50 samples at the beginning and end to remove)
    padding_samples = 50;
    
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
        eeg_signals(~isfinite(eeg_signals)) = 0;  % Replace NaNs and Infs with 0
        
        % Normalize the signals to have mean 0 and standard deviation 1
        eeg_signals = (eeg_signals - mean(eeg_signals, 1)) ./ std(eeg_signals, 0, 1);
    
        % Apply the notch filter to each channel
        eeg_notched = filtfilt(b_notch, a_notch, eeg_signals);
    
        % Apply the band-pass filter to each channel
        eeg_filtered_bp = filtfilt(b_bpf, a_bpf, eeg_notched);
    
        % Apply the low-pass filter to each channel
        eeg_filtered = filtfilt(b_lp, a_lp, eeg_filtered_bp);
    
        % Remove padding from the beginning and end
        eeg_filtered = eeg_filtered(padding_samples+1:end-padding_samples, :);
        
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
end
