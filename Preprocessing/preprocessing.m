function preprocessing()
%PREPROCESSING Summary of this function goes here
%   Detailed explanation goes here

% emg_signals = preprocessing_emg()
% eeg_signals = preprocessing_eeg()

%% Retrieving segnali
% Define paths and constants
eeg_data_directory = 'unsegmented_filtered_EEG_data\'; % Path to the directory containing EEG data
emg_data_directory = 'unsegmented_filtered_EMG_data\'; % Path to the directory containing EMG data
eeg_file_list = dir(fullfile(eeg_data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories
emg_file_list = dir(fullfile(emg_data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories

% Assuming EEG and EMG have the same file structure and naming
sampling_frequency_eeg = 250; % Hz
sampling_frequency_emg = 200; % Hz

% Create directory for saving features
cmc_features_directory = 'CMC_features';
if ~exist(cmc_features_directory, 'dir')
    mkdir(cmc_features_directory);
end

for file_idx = 1:numel(emg_file_list)
    % Load the filtered EEG and EMG data
    eeg_file_name = eeg_file_list(file_idx).name;
    emg_file_name = emg_file_list(file_idx).name;
    
    eeg_file_path = fullfile(eeg_file_list(file_idx).folder, eeg_file_name);
    emg_file_path = fullfile(emg_file_list(file_idx).folder, emg_file_name);
    
    eeg_data = load(eeg_file_path);
    emg_data = load(emg_file_path);

    % Access the filtered EEG and EMG data
    eeg = eeg_data.eeg_filtered;
    emg = emg_data.emg_filtered;

    % Definisci la lunghezz a della finestra per la RMS (in campioni)
    window_length_rms = 250; % ad esempio, 1 secondo per un fs di 250 Hz

    for ch = 1:size(emg, 2)
        emg = abs(emg);
        rest_on = size(emg,1) / sampling_frequency_emg - 3;
%         emg = emg()

%         first_window_size = 800;
%         second_window_size = 200;
%         emg_signals_smoothed = zeros(size(emg,1), size(emg,2));
%         for i = 1:size(emg,2)
%             emg_signals_smoothed(:,i) = moving_average_array(emg(:,i), first_window_size);
%             emg_signals_smoothed(:,i) = moving_average_array(emg_signals_smoothed(:,i), second_window_size);
%         end
% 
%         emg_rms = zeros(size(emg_signals_smoothed));
%         % Applica la RMS a ciascun canale del segnale filtrato
%             
%         emg_rms(:, ch) = sqrt(movmean(emg_signals_smoothed(:, ch).^2, window_length_rms));
% 

%         % Applicazione Threshold
%         Th = (max(emg_rms) - min(emg_rms)) / 3 + min(emg_rms);
        
        % Find intervals above the threshold
%         above_threshold = emg_rms > Th;
%         segments = find_segments(above_threshold);
    
        t = (0:size(emg,1)-1) / sampling_frequency_emg;
        segments_time = size(emg,1) / sampling_frequency_emg;

        figure;
        plot(t, emg(:, ch))
        hold on;
        for i = 1:size(segments_time, 1)
            x = [rest_on, rest_on];
            line(x, ylim, 'Color', 'r', 'LineStyle', '--');
        end
        xlabel('Time (s)');
        ylabel('Amplitude (\muV)');
        title('EMG Signal with Segmentation Points');
        hold off;

    end

    % Duration of each segment in seconds
    DUR = 2;  % You can adjust this duration as needed
    segment_samples = DUR * sampling_frequency_emg;
    
    % Segmentazione
    for seg_idx = 1:size(segments, 1)
        start_sample = max(segments(seg_idx, 1) - segment_samples / 2, 1);
        end_sample = min(segments(seg_idx, 2) + segment_samples / 2, length(emg));

        emg_segment = emg(start_sample:end_sample, :);
        
        % Corresponding EEG segment
        eeg_start_sample = round(start_sample * sampling_frequency_eeg / sampling_frequency_emg);
        eeg_end_sample = round(end_sample * sampling_frequency_eeg / sampling_frequency_emg);
        eeg_start_sample = max(eeg_start_sample, 1);
        eeg_end_sample = min(eeg_end_sample, size(eeg, 1));
        
        eeg_segment = eeg(eeg_start_sample:eeg_end_sample, :);

        % Compute CMC for the segment
        [S_x, S_y, S_xy, f, fs] = compute_power_spectrum(eeg_segment, emg_segment, sampling_frequency_eeg, sampling_frequency_emg);
        
        num_channels = size(eeg, 2); % Number of channels
    end
    % Calculate average power spectra for the current channel
    S_x_avg = mean(S_x(:, ch, :), 3);
    S_y_avg = mean(S_y(:, ch, :), 3);
    S_xy_avg = mean(S_xy(:, ch, :), 3);

    % Initialize a matrix to store CMC for each channel
    CMC = zeros(size(S_xy, 1), num_channels);
    
    for ch = 1:num_channels
        
        % Calculate the CMC for the current channel
        squared_CMC = (abs(S_xy(:, ch)).^2) ./ (S_x(:, ch) .* S_y(:, ch));

        % Normalize the CMC to be between 0 and 1
        CMC_normalized = sqrt(squared_CMC); % / max(squared_CMC);
        
        % Handle NaN in CMC_normalized
        if any(isnan(CMC_normalized))
            CMC_normalized(isnan(CMC_normalized)) = 0;
        end

        % Calculate the CMC in the time domain
        CMC_time = ifft(CMC_normalized);

        % Ensure CMC_time is real
        CMC(:, ch) = real(CMC_time);
    end
    
    % Perform feature extraction
    [features, parameters] = universal_feature_extraction(CMC, fs, 'emg');
    
    % Construct save filename
    [~, base_name, ~] = fileparts(eeg_file_name);
    save_filename = fullfile(cmc_features_directory, [base_name '_features.mat']);

    % Save the CMC features
    save(save_filename, 'features', 'parameters');

    % Print debug information
    fprintf('Extracted CMC features for all channels and saved to: %s\n', save_filename);
    end
end
