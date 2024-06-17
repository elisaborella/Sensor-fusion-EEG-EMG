% Directory containing the files
base_dir = 'EMG_features\';  % Adjust this path as needed

% List all files in the directory and subdirectories
file_list = dir(fullfile(base_dir, '**', '*'));

% Iterate through each file
for file_idx = 1:numel(file_list)
    % Check if the file name contains 'filtered'
    if contains(file_list(file_idx).name, 'filtered')
        % Construct the full file path
        file_path = fullfile(file_list(file_idx).folder, file_list(file_idx).name);
        
        % Delete the file
        delete(file_path);
    end
end
