clear all

%%
preprocessing()
%%

% % %% TEST
emg_data = load("unsegmented_filtered_EMG_data\S1_R1_G7\S1_R1_G7.mat");
% eeg_data = load("unsegmented_filtered_EEG_data\S1_R3_G6\S1_R3_G6.mat");
% % 
emg_filtered_signal = emg_data.emg_filtered;
fs_emg = emg_data.fs_emg;
% eeg_signal = eeg_data.eeg_filtered;
% fs_eeg = eeg_data.fs_eeg;
% % 
%plotter(emg_signal, fs_emg, "EMG filtered signal");
% plotter(eeg_signal, fs_eeg, "EEG signal");
%%
[S_x, S_y, S_xy, fs] = compute_power_spectrum(eeg_signal, emg_signal, fs_eeg, fs_emg);
plot_power_spectrum(S_x, S_y, S_xy, fs_eeg);

CMC = zeros(size(S_xy,1));
for i = 1 :size(S_xy,2)
    S_x_avg = mean(S_x(:, i, :), 3);
    S_y_avg = mean(S_y(:, i, :), 3);
    S_xy_avg = mean(S_xy(:, i, :), 3);
    % Calcolare il CMC
    squared_CMC = (abs(S_xy(:,i).^2) ./ (S_x(:,i) .* S_y(:,i)));
    
    % Normalizzare il CMC per ottenere valori tra 0 e 1
    CMC(:,i) = squared_CMC / max(squared_CMC);
    
    % Visualizzare il CMC
    figure;
    plot(fs, CMC(:,i));
    xlabel('Frequency (Hz)');
    ylabel('CMC');
    title(sprintf('Magnitude Square Coherence (CMC) - Channel %d', i));
    grid on;
    hold on;
    
    % Definire le sub-bande di frequenza
    sub_bands = [6 8; 8 12; 13 20; 20 30; 13 30; 30 60; 60 80; 30 80];
    
    % Aggiungere le linee verticali rosse per ogni sub-banda
    for j = 1:size(sub_bands, 1)
        xline(sub_bands(j, 1), 'r--');
        xline(sub_bands(j, 2), 'r--');
    end
    
    % Legenda delle sub-bande
    legend_labels = {'Low-α', 'α', 'Low-β', 'High-β', 'β', 'Low-γ', 'High-γ', 'γ'};
    for j = 1:size(sub_bands, 1)
        % Centrare il testo della legenda tra le linee delle sub-bande
        text(mean(sub_bands(j, :)), max(CMC(:,i))*0.9, legend_labels{j}, 'Color', 'r', 'HorizontalAlignment', 'center');
    end
    
    hold off;
end

%% EMG FEATURES EXTRACTION
% Define paths and constants
data_directory = 'unsegmented_filtered_EMG_data\'; % Path to the directory containing subject data
file_list = dir(fullfile(data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories
sampling_frequency_emg = 200; % Hz

% Create directory for saving features
features_directory = 'EMG_features';
if ~exist(features_directory, 'dir')
    mkdir(features_directory);
end

for file_idx = 1:numel(file_list)
    % Load the filtered EMG data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    emg_data = load(file_path);

    % Access the filtered EMG data
    emg_filtered = emg_data.emg_filtered;

    % Perform feature extraction
    [features, parameters] = universal_feature_extraction(emg_filtered, sampling_frequency_emg, 'emg');

    % Construct save filename
    [~, base_name, ~] = fileparts(file_name);
    save_filename = fullfile(features_directory, [base_name '_features.mat']);

    % Save the extracted features
    save(save_filename, 'features', 'parameters');

    % Print debug information
    fprintf('Extracted features and saved to: %s\n', save_filename);
end



%% EEG FEATURES EXTRACTION
% Define paths and constants
data_directory = 'unsegmented_filtered_EEG_data\'; % Path to the directory containing subject data
file_list = dir(fullfile(data_directory, '**', '*.mat')); % List all filtered segment .mat files in subdirectories
sampling_frequency_eeg = 250; % Hz

% Create directory for saving features
features_directory = 'EEG_features';
if ~exist(features_directory, 'dir')
    mkdir(features_directory);
end

% Iterate through each filtered file
for file_idx = 1:numel(file_list)
    % Load the filtered EEG data
    file_name = file_list(file_idx).name;
    file_path = fullfile(file_list(file_idx).folder, file_name);
    eeg_data = load(file_path);

    % Access the filtered EEG data
    eeg_filtered = eeg_data.eeg_filtered;

    % Perform feature extraction
    [features, parameters] = universal_feature_extraction(eeg_filtered, sampling_frequency_eeg, 'eeg');

    % Construct save filename
    [~, base_name, ~] = fileparts(file_name);
    save_filename = fullfile(features_directory, [base_name '_features.mat']);

    % Save the extracted features
    save(save_filename, 'features', 'parameters');

    % Print debug information
    fprintf('Extracted features and saved to: %s\n', save_filename);
end

%% CMC features extraction
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
    eeg_filtered = eeg_data.eeg_filtered;
    emg_filtered = emg_data.emg_filtered;

    % Perform CMC calculation
    [S_x, S_y, S_xy, f, fs] = compute_power_spectrum(eeg_filtered, emg_filtered, sampling_frequency_eeg, sampling_frequency_emg);
    
    num_channels = size(eeg_filtered, 2); % Number of channels

    % Initialize a cell array to store CMC for each channel
    CMC = zeros(size(S_xy,1), num_channels);

    for ch = 1:num_channels
        % Calculate average power spectra for the current channel
        S_x_avg = mean(S_x(:, ch, :), 3);
        S_y_avg = mean(S_y(:, ch, :), 3);
        S_xy_avg = mean(S_xy(:, ch, :), 3);

        % Calculate the CMC for the current channel
        squared_CMC = (abs(S_xy(:,ch)).^2) ./ (S_x(:,ch) .* S_y(:,ch));

        % Normalize the CMC to be between 0 and 1
        CMC_normalized = sqrt(squared_CMC);% / max(squared_CMC);
        
        % Handle NaN in CMC_normalized
        if any(isnan(CMC_normalized))
            % Replace NaN with zeros or handle as appropriate
            CMC_normalized(isnan(CMC_normalized)) = 0;
        end

        % Calculate the CMC in the time domain
        CMC_time = ifft(CMC_normalized);

        % Ensure CMC_time is real
        CMC(:,ch) = real(CMC_time);
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
