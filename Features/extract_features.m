function extract_features(data_directory, features_directory, type_of_signal)
    % EXTRACT_FEATURES Extracts features from filtered segment data and saves them
    %   data_directory: Directory containing filtered segment .mat files
    %   features_directory: Directory to save extracted features
    %   type_of_signal: Type of signal ('emg', 'eeg', 'cmc')

    % List all filtered segment .mat files in subdirectories
    file_list = dir(fullfile(data_directory, '**', '*.mat')); 
    
    % Create directory for saving features
    if ~exist(features_directory, 'dir')
        mkdir(features_directory);
    end
    
    for file_idx = 1:numel(file_list)
        % Load the filtered data
        file_name = file_list(file_idx).name;
        file_path = fullfile(file_list(file_idx).folder, file_name);
        data = load(file_path);
    
        % Access the filtered data based on type_of_signal
        switch type_of_signal
            case 'emg'
                filtered_data = data.emg_filtered;
                fs_data = data.fs_emg;
            case 'eeg'
                filtered_data = data.eeg_filtered;
                fs_data = data.fs_eeg;
            case 'cmc'
                % Perform CMC calculation
                [S_x, S_y, S_xy, ~, fs_data] = compute_power_spectrum(data.eeg_filtered, data.emg_filtered, data.fs_eeg, data.fs_emg);
                CMC = compute_cmc(S_x, S_y, S_xy);
                
                % Perform feature extraction on CMC
                [features, parameters] = universal_feature_extraction(CMC, fs_data, 'cmc');
        end
    
        % Perform feature extraction on EMG or EEG
        if strcmp(type_of_signal, 'emg') || strcmp(type_of_signal, 'eeg')
            [features, parameters] = universal_feature_extraction(filtered_data, fs_data, type_of_signal);
        end
    
        % Construct save filename
        [~, base_name, ~] = fileparts(file_name);
        save_filename = fullfile(features_directory, [base_name '_features.mat']);
    
        % Save the extracted features
        save(save_filename, 'features', 'parameters');
    
        % Print debug information
        fprintf('Extracted features and saved to: %s\n', save_filename);
    end
end
