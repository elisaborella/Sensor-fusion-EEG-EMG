function segments = segmentation(signal, fs,window_length_ms, overlap_percentage, filtered_data_dir)
%SEGMENTATION Summary of this function goes here
%   Detailed explanation goes here

% Create the external directory if it doesn't exist
if ~exist(filtered_data_dir, 'dir')
    mkdir(filtered_data_dir);
end

window_length_samples = round(window_length_ms / 1000 * fs);
stride_samples = round(window_length_samples * (1 - overlap_percentage / 100));
    
    
% Segment the filtered signals
num_samples = size(signal, 1);
num_channels = size(signal, 2);
segments = {};

start_idx = 1;
while start_idx + window_length_samples - 1 <= num_samples
    end_idx = start_idx + window_length_samples - 1;
    segment = signal(start_idx:end_idx, :);
    segments{end + 1} = segment;
    start_idx = start_idx + stride_samples;
end

