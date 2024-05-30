% Define paths and constants
% data_directory = 'BMIS_EMG_DATA\data\mat_data'; % Path to the directory containing subject data
% subjects = dir(data_directory); % List of subject directories
% subjects = subjects([subjects.isdir]); % Remove non-directory entries
% subjects = {subjects.name}; % Extract directory names
% subjects = subjects(3:end); % Remove '.' and '..' directories
% gestures = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'}; % List of gesture names
% repetitions = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}; % List of repetition names
% sampling_frequency_emg = 200; % Hz
% 
% % Loop over subjects
% for s = 1:length(subjects)
%     subject_directory = fullfile(data_directory, subjects{s});
% 
%     % Extract subject number from directory name using regular expression
%     subject_num_match = regexp(subjects{s}, '\d+', 'match'); % Extract numeric part from the directory name
%     subject_num = str2double(subject_num_match{1}); % Convert the extracted numeric part to a number
% 
%     % Loop over repetitions
%     for r = 1:length(repetitions)
% 
%         % Loop over gestures
%         for g = 1:length(gestures)
%             % Construct the filename using the correct format
%             emg_filename = fullfile(subject_directory, sprintf('S%d_%s_%s.mat', subject_num, repetitions{r}, gestures{g}));
% 
%             % Check if the file exists
%             if exist(emg_filename, 'file')
%                 % Load EMG data
%                 emg_data = load(emg_filename);
% 
%                 % Access the EMG data from the loaded file
%                 dataset = emg_data.data;
% 
%                 % Perform feature extraction
%                 [features, parameters] = universal_feature_extraction(dataset, sampling_frequency_emg, 'emg');
% 
%                 % Now you can use the extracted features and parameters
%                 % For example, you can save the features for each subject, repetition, and task
%                 save_filename = fullfile('EMG_features', sprintf('S%d_%s_%s_features.mat', subject_num, repetitions{r}, gestures{g}));
%                 save(save_filename, 'features', 'parameters');
%             else
%                 fprintf('File %s not found.\n', emg_filename);
%             end
%         end
%     end
% end
% 
% 
% data_directory = 'BMIS_EEG_DATA\data\mat_data'; % Path to the directory containing subject data
% subjects = dir(data_directory); % List of subject directories
% subjects = subjects([subjects.isdir]); % Remove non-directory entries
% subjects = {subjects.name}; % Extract directory names
% subjects = subjects(3:end); % Remove '.' and '..' directories
% gestures = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'}; % List of gesture names
% repetitions = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}; % List of repetition names
% sampling_frequency_eeg = 250; % Hz
% 
% % Loop over subjects
% for s = 1:length(subjects)
%     subject_directory = fullfile(data_directory, subjects{s});
% 
%     % Extract subject number from directory name using regular expression
%     subject_num_match = regexp(subjects{s}, '\d+', 'match'); % Extract numeric part from the directory name
%     subject_num = str2double(subject_num_match{1}); % Convert the extracted numeric part to a number
% 
%     % Loop over repetitions
%     for r = 1:length(repetitions)
% 
%         % Loop over gestures
%         for g = 1:length(gestures)
%             % Construct the filename using the correct format
%             eeg_filename = fullfile(subject_directory, sprintf('S%d_%s_%s.mat', subject_num, repetitions{r}, gestures{g}));
% 
%             % Check if the file exists
%             if exist(eeg_filename, 'file')
%                 % Load EEG data
%                 eeg_data = load(eeg_filename);
% 
%                 % Access the EEG data from the loaded file and transpose
%                 dataset = permute(eeg_data.data, [2, 1, 3]); % Transpose data matrix
% 
%                 % Perform feature extraction
%                 [features, parameters] = universal_feature_extraction(dataset, sampling_frequency_eeg, 'eeg');
% 
%                 % Now you can use the extracted features and parameters
%                 % For example, you can save the features for each subject, repetition, and task
%                 save_filename = fullfile('EEG_features', sprintf('S%d_%s_%s_features.mat', subject_num, repetitions{r}, gestures{g}));
%                 save(save_filename, 'features', 'parameters');
%             else
%                 fprintf('File %s not found.\n', eeg_filename);
%             end
%         end
%     end
% end

%%
clear all

%% EMG signals
% Load the data
data = load('BMIS_EMG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');

% Access the EMG signals
emg_signals = data.data;

% Sampling frequency
fs_emg = 200;

% Time vector
t = (0:size(emg_signals, 1)-1) / fs_emg;

% Plot some sample EMG signals
figure;
for i = 1:min(size(emg_signals, 2), 8) % Plot up to the first 4 channels
    subplot(min(size(emg_signals, 2), 8), 1, i);
    plot(t, emg_signals(:, i));
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['EMG Signal Channel ', num2str(i)]);
    grid on;
end

%% EEG Signals

% Load the data
data = load('BMIS_EEG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');

% Access the EEG signals
eeg_signals = permute(data.data, [2, 1, 3]);

% Sampling frequency
fs_eeg = 250;

% Time vector
t = (0:size(eeg_signals, 1)-1) / fs_eeg;

%% Plot time domain

% Plot some sample EEG signals
figure;
% for i = 1:min(size(eeg_signals, 2), 8) % Plot up to the first 4 channels
%     subplot(min(size(eeg_s_one_sec, 2), 8), 1, i);
plot(t, eeg_signals(:, 1));
xlabel('Time (s)');
ylabel('Amplitude');
title('EEG Signal ');
%     title(['EEG Signal Channel ', num2str(i)]);
grid on;
% end

%% Preprocessing
% FcutHPF = 0.5; % High-pass filter cut-off frequency in Hz
% HdHPF = HighPassFilter(fs_eeg, FcutHPF);
% eeg_filtered = filtfilthd(HdHPF, eeg_signals(:,1)); % Apply high-pass filter in time domain
% 
% 

%% Frequency domain analysis
% Apply FFT over the filtered signal
Fcut1BPF = 4;
Fcut2BPF = 55;
BW = 4;       % Bandwidth
HdBPF = BandPassFilter(fs_eeg, Fcut1BPF, Fcut2BPF, BW);
eeg_filtered = filtfilthd(HdBPF, eeg_signals(:,1));

S = fft(eeg_filtered);
NFFT = length(S);

% Define frequency resolution and frequency vector
F = fs_eeg / NFFT;
f = F * (0:NFFT-1);

% Calculate power spectral density (PSD)
P = abs(S).^2 / NFFT; % Normalize by NFFT to get power per Hz

% Plot the PSD
figure;
plot(f, P);
axis([4 100 0 max(P)])     % Define the visualization range axis([xmin xmax ymin ymax])
xlabel('Frequency (Hz)');
ylabel('Power (uV^2/Hz)');
title('EEG Signal Frequency Domain');
grid on;


%%
figure;
plot(t, eeg_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
title('EEG Signal ');
grid on;

%%
% 
% num_features = size(features, 1);
% 
% disp(['Number of features: ', num2str(num_features)]);
% 
% % Choose a specific feature index (e.g., 1 for MIN)
% feature_index = 2;
% 
% % Extract the feature values for the chosen feature index
% feature_values = squeeze(features(feature_index, :, :)); % NCH x NTRIALS
% 
% % Calculate statistics
% mean_feature = mean(feature_values, 2); % Mean across trials for each channel
% median_feature = median(feature_values, 2); % Median across trials for each channel
% std_feature = std(feature_values, 0, 2); % Standard deviation across trials for each 6channel
% range_feature = range(feature_values, 2); % Range across trials for each channel
% 
% % Display the statistics
% disp('Statistics for the chosen feature:')
% disp(['Mean: ', num2str(mean_feature')])
% disp(['Median: ', num2str(median_feature')])
% disp(['Standard Deviation: ', num2str(std_feature')])
% disp(['Range: ', num2str(range_feature')])

%% CMC
DUR = 4; % Duration of segment (s)
[S_x, S_y, S_xy, Fs] = compute_power_spectrum(double(eeg_filtered), double(emg_signals), DUR, fs_eeg, fs_emg);


% Plotting
figure;

% Plot dello spettro di potenza EEG (S_x)
subplot(3, 1, 1);
plot((0:size(S_x,1)-1)*(fs_eeg/DUR), S_x(:,1,1)); % Assumiamo il primo segmento e il primo canale
title('Spettro di Potenza EEG (S_x)');
xlabel('Frequenza (Hz)');
ylabel('Potenza');

% Plot dello spettro di potenza EMG (S_y)
subplot(3, 1, 2);
plot((0:size(S_y,1)-1)*(fs_emg/DUR), S_y(:,1,1)); % Assumiamo il primo segmento e il primo canale
title('Spettro di Potenza EMG (S_y)');
xlabel('Frequenza (Hz)');
ylabel('Potenza');

% Plot dello spettro di potenza incrociato (S_xy)
subplot(3, 1, 3);
plot((0:size(S_xy,1)-1)*(Fs/DUR), S_xy(:,1,1)); % Assumiamo il primo segmento e il primo canale
title('Spettro di Potenza Incrociato (S_xy)');
xlabel('Frequenza (Hz)');
ylabel('Potenza');