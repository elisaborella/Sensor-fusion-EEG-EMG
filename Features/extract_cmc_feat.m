function extract_cmc_feat(eeg_data_directory, emg_data_directory, features_directory)
    % EXTRACT_CMC_FEAT Extracts CMC features from EEG and EMG data files
    %   eeg_data_directory: Directory containing EEG data files
    %   emg_data_directory: Directory containing EMG data files
    %   features_directory: Directory to save extracted features
    
    % Define paths and constants
    eeg_file_list = dir(fullfile(eeg_data_directory, '**', '*.mat')); % List all EEG .mat files
    emg_file_list = dir(fullfile(emg_data_directory, '**', '*.mat')); % List all EMG .mat files
    
    % Create directory for saving features if it doesn't exist
    if ~exist(features_directory, 'dir')
        mkdir(features_directory);
    end
    
    % Extract subject and trial information from EEG file list
    subject_trial_files = extract_trials(emg_file_list);
    
    % Process each subject and trial
    subject_keys = keys(subject_trial_files);
    for k = 1:numel(subject_keys)
        key = subject_keys{k};
        file_indices = subject_trial_files(key);
        
        all_S_x = [];
        all_S_y = [];
        all_S_xy = [];
    
        % Process each file index for the current subject and trial
        for file_idx = file_indices
            file_index = file_idx{1};
    
            % Extract file names and paths
            eeg_file_name = eeg_file_list(file_index).name;
            emg_file_name = emg_file_list(file_index).name;
            
            eeg_file_path = fullfile(eeg_file_list(file_index).folder, eeg_file_name);
            emg_file_path = fullfile(emg_file_list(file_index).folder, emg_file_name);
    
            % Load EEG and EMG data
            eeg_data = load(eeg_file_path);
            emg_data = load(emg_file_path);
    
            % Access the filtered EEG and EMG data
            eeg = eeg_data.eeg_filtered;
            fs_eeg = eeg_data.fs_eeg;
            emg = emg_data.emg_filtered;
            fs_emg = emg_data.fs_emg;
    
            % Perform CMC calculation
            [S_x, S_y, S_xy, ~, fs] = compute_power_spectrum(eeg, emg, fs_eeg, fs_emg);
            
            % Accumulate power spectra
            all_S_x = cat(3, all_S_x, S_x);
            all_S_y = cat(3, all_S_y, S_y);
            all_S_xy = cat(3, all_S_xy, S_xy);
        end
    
        % Calculate average power spectra
        S_x_avg = mean(all_S_x, 3);
        S_y_avg = mean(all_S_y, 3);
        S_xy_avg = mean(all_S_xy, 3);
    
        % Compute CMC for each channel
        CMC = compute_cmc(S_x_avg, S_y_avg, S_xy_avg);
    
        % Perform feature extraction
        [features, parameters] = universal_feature_extraction(CMC, fs, 'emg');
    
        % Construct save filename
        save_filename = fullfile(features_directory, [key '_features.mat']);
    
        % Save the CMC features
        save(save_filename, 'features', 'parameters');
    
        % Print debug information
        fprintf('Extracted CMC features for %s and saved to: %s\n', key, save_filename);
    end
end