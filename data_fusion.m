clear;
clc;

% Parameters
fs_eeg = 250;
fs_emg = 200;
DUR = 2;
channels_to_process = 1; % Adjust as needed

% Directories with EEG and EMG data files
eeg_directory = 'filtered_EEG_data';
emg_directory = 'filtered_EMG_data';
output_file = 'CMC_results.mat'; % File to save CMC results

% Get a list of all EEG files
eeg_files = dir(fullfile(eeg_directory, 'S*_R*_G*', '*_filtered.mat'));
num_files = numel(eeg_files);

% Initialize variables to store CMC results
CMC_results = struct();
CMC_values = zeros(num_files, 1);

% Iterate over each EEG file
for i = 1:num_files
    eeg_filename = fullfile(eeg_files(i).folder, eeg_files(i).name);
    
    % Extract subject, run, and gesture from EEG filename
    [~, eeg_filename_short, ~] = fileparts(eeg_filename);
    fprintf('Processing file %d/%d:\n', i, num_files);
    fprintf('EEG file: %s\n', eeg_filename);
    
    % Extract subject, run, and gesture from filename
    parts = strsplit(eeg_filename_short, '_');
    
    % Check if the parts are as expected
    if numel(parts) < 3
        error('Unexpected filename format: %s', eeg_filename_short);
    end
    
    % Extract subject, run, and gesture
    subject = parts{1}; % Keep 'S' in the subject
    run = parts{2};
    gesture = parts{3};

    % Construct corresponding EMG filename
    emg_filename = fullfile(emg_directory, [subject '_' run '_' gesture], [subject '_' run '_' gesture '_filtered.mat']);
    
    % Print debug information
    fprintf('EMG file: %s\n', emg_filename);
    
    try
        CMC_value = compute_CMC(eeg_filename, emg_filename, fs_eeg, fs_emg, DUR, channels_to_process);
        % Store CMC value
        CMC_results(i).eeg_filename = eeg_filename;
        CMC_results(i).cmcs = CMC_value;
        % Print CMC value
        fprintf('CMC value: %.4f\n\n', CMC_value);
        % Store CMC value
        CMC_values(i) = CMC_value;
    catch ME
        fprintf('Error processing %s and %s:\n%s\n', eeg_filename, emg_filename, ME.message);
        continue;
    end
end

% Save CMC results
save(output_file, 'CMC_results');

fprintf('CMC values saved to %s\n', output_file);

% Function to compute CMC
function CMC_value = compute_CMC(eeg_filename, emg_filename, fs_eeg, fs_emg, DUR, channels_to_process)
    try
        eeg_data = load(eeg_filename);
        var_name = fieldnames(eeg_data);
        eeg_signals = double(eeg_data.(var_name{1}));
        eeg_signals = permute(eeg_signals, [2, 1]); % Transpose to have time along rows
        
        % Load the EMG data
        emg_data = load(emg_filename);
        var_name = fieldnames(emg_data);
        emg_signals = double(emg_data.(var_name{1}));
        emg_signals = permute(emg_signals, [2, 1]); % Transpose to have time along rows
        
        % Sampling frequency
        fs_eeg = 250;
        fs_emg = 200;
        
        % Duration of segment in seconds
        DUR = 2;
        
        % Select specific channels to process
        channels_to_process = 1; % Adjust as needed
        
        % Ensure signals have the same length
        min_length = min(size(eeg_signals, 2), size(emg_signals, 2));
        eeg_signals = eeg_signals(:, 1:min_length);
        emg_signals = emg_signals(:, 1:min_length);
        
        % Compute the power spectra and cross power spectrum using FFT directly
        [S_x, f_x] = compute_power_spectrum(eeg_signals(channels_to_process,:), DUR, fs_eeg);
        [S_y, f_y] = compute_power_spectrum(emg_signals(channels_to_process,:), DUR, fs_emg);
        [S_xy, f_xy] = compute_cross_power_spectrum(eeg_signals(channels_to_process,:), emg_signals(channels_to_process, :), fs_eeg, fs_emg);
        
        % Normalize the power spectra
        S_x = S_x / sum(S_x); % Normalize S_x
        S_y = S_y / sum(S_y); % Normalize S_y
        S_xy = abs(S_xy).^2 / sum(abs(S_xy).^2); % Normalize |S_xy|^2
        
        % Ensure the power spectra and cross power spectrum have the same length
        min_length_spectra = min([length(S_x), length(S_y), length(S_xy)]);
        S_x = S_x(1:min_length_spectra);
        S_y = S_y(1:min_length_spectra);
        S_xy = S_xy(1:min_length_spectra);
        
        % Debugging information
        fprintf('Max S_x: %.4f, Max S_y: %.4f, Max S_xy: %.4f\n', max(S_x), max(S_y), max(S_xy));
        
        % Compute the Coherence Magnitude Coefficient (CMC)
        CMC = abs(S_xy).^2 ./ (S_x .* S_y);
        CMC = mean(CMC(CMC <= 1)); % Ensure CMC values are between 0 and 1

        % Return the CMC value
        CMC_value = CMC;
    catch ME
        fprintf('Error processing %s and %s:\n', eeg_filename, emg_filename);
        rethrow(ME);
    end
end

function [S, f] = compute_power_spectrum(signal, DUR, fs)
    % Compute power spectrum using FFT
    L = size(signal, 2);
    NFFT = 2^nextpow2(L);
    Y = fft(signal, NFFT) / L;
    f = fs / 2 * linspace(0, 1, NFFT/2 + 1);
    S = 2 * abs(Y(1:NFFT/2 + 1)).^2; % Power spectrum
end

function [S_xy, f_xy] = compute_cross_power_spectrum(signal1, signal2, fs1, fs2)
    % Compute cross power spectrum using FFT
    L1 = size(signal1, 2);
    L2 = size(signal2, 2);
    NFFT = 2^nextpow2(min(L1, L2));
    Y1 = fft(signal1, NFFT) / L1;
    Y2 = fft(signal2, NFFT) / L2;
    f_xy = min(fs1, fs2) / 2 * linspace(0, 1, NFFT/2 + 1);
    S_xy = Y1(1:NFFT/2 + 1) .* conj(Y2(1:NFFT/2 + 1));
end


