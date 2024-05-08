% % Define paths and constants
% data_directory = 'BMIS_EMG_DATA\data\mat_data'; % Path to the directory containing subject data
% subjects = dir(data_directory); % List of subject directories
% subjects = subjects([subjects.isdir]); % Remove non-directory entries
% subjects = {subjects.name}; % Extract directory names
% subjects = subjects(3:end); % Remove '.' and '..' directories
% gestures = {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'}; % List of gesture names
% repetitions = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}; % List of repetition names
% sampling_frequency_emg = 200; % Hz
% sampling_frequency_eeg = 250; % Hz
% 
% % Loop over subjects
% for s = 1:length(subjects)
%     subject_directory = fullfile(data_directory, subjects{s});
% 
%     % Loop over gestures
%     for g = 1:length(gestures)
% 
%         % Loop over repetitions
%         for r = 1:length(repetitions)
%             % Load EMG data
%             emg_filename = fullfile(subject_directory, sprintf('S%s_%s_%s.mat', extractAfter(subjects{s},'_'), repetitions{r}, gestures{g}));
%             if exist(emg_filename, 'file')
%                 emg_data = load(emg_filename);
% 
%                 % Preprocess EMG data if necessary
% 
%                 % Extract features from EMG data
%                 [emg_features, emg_parameters] = universal_feature_extraction(emg_data, sampling_frequency_emg, 'emg');
% 
%                 % Now you can use the extracted features for machine learning
% 
%                 % For example, you can save the features for each subject, repetition, and gesture
%                 save_filename = fullfile('EMG_features/', sprintf('%s_%s_%s_features.mat', subjects{s}, repetitions{r}, gestures{g}));
%                 save(save_filename, 'emg_features', 'emg_parameters');
%             else
%                 fprintf('File %s not found.\n', emg_filename);
%             end
%         end
%     end
% end

% Load the data
data = load('BMIS_EMG_DATA\data\mat_data\subject_1\S1_R1_G1.mat');

% Access the EMG signals
emg_signals = data.data;

% Sampling frequency
fs = 200; % Assuming a sampling frequency of 200 Hz

% Time vector
t = (0:size(emg_signals, 1)-1) / fs;

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

