function subject_trial_files = extract_trials(file_list)
    % EXTRACT_TRIALS Organizes files by subject and trial
    %   file_list: List of files to organize
    
    % Initialize container for subject and trial files
    subject_trial_files = containers.Map();
    
    % Iterate through each file in the file list
    for file_idx = 1:numel(file_list)
        file_name = file_list(file_idx).name;
    
        % Extract subject and trial information from the file name
        tokens = regexp(file_name, 'S(\d+)_R(\d+)_G(\d+)', 'tokens');
        if isempty(tokens)
            continue;  % Skip files that do not match the expected pattern
        end
        subject_id = str2double(tokens{1}{1});
        grasp_id = str2double(tokens{1}{3});
    
        % Create a key for the subject and grasp combination
        key = sprintf('S%d_G%d', subject_id, grasp_id);
        
        % Check if the key already exists in the map
        if ~isKey(subject_trial_files, key)
            subject_trial_files(key) = {file_idx};  % Initialize with a cell array if key is new
        else
            subject_trial_files(key) = [subject_trial_files(key), file_idx];  % Append to existing list if key exists
        end
    end
end
