function CMC = compute_cmc(S_x, S_y, S_xy)
    % COMPUTE_CMC Calculates the Coherence Magnitude Coefficient (CMC) from power spectra
    %   S_x: Power spectrum of the first signal (columns represent channels)
    %   S_y: Power spectrum of the second signal (columns represent channels)
    %   S_xy: Cross power spectrum between corresponding channels of S_x and S_y
    %   CMC: Coherence Magnitude Coefficient matrix (each column corresponds to a channel)

    % Initialize CMC matrix
    CMC = zeros(size(S_xy, 1), size(S_xy, 2));
    
    % Number of channels
    num_channels = size(S_x, 2); 
    
    % Calculate CMC for each channel
    for ch = 1:num_channels
        % Calculate the squared CMC
        squared_CMC = (abs(S_xy(:, ch)).^2) ./ (S_x(:, ch) .* S_y(:, ch));
    
        % Normalize the CMC to be between 0 and 1
        CMC_normalized = sqrt(squared_CMC);
    
        % Handle NaN values in CMC_normalized
        if any(isnan(CMC_normalized))
            CMC_normalized(isnan(CMC_normalized)) = 0;
        end
    
        % Calculate the CMC in the time domain using inverse FFT
        CMC_time = ifft(CMC_normalized);
    
        % Ensure CMC_time is real-valued
        CMC(:, ch) = real(CMC_time);
    end
end
