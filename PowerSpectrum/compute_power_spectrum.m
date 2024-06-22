function [S_x, S_y, S_xy, f, fs_cross] = compute_power_spectrum(eeg_signals, emg_signals, Fs_eeg, Fs_emg)
    % COMPUTE_POWER_SPECTRUM Computes the power spectrum of EEG channels and the cross-power spectrum
    %   eeg_signals: matrix where each column represents an EEG channel
    %   emg_signals: matrix where each column represents an EMG channel
    %   Fs_eeg: sampling frequency of the EEG signal (in Hz)
    %   Fs_emg: sampling frequency of the EMG signal (in Hz)
    %   S_x: matrix where each column represents the power spectrum of an EEG channel
    %   S_y: matrix where each column represents the power spectrum of an EMG channel
    %   S_xy: matrix where each column represents the cross-power spectrum between corresponding channels
    %   f: vector of frequencies
    %   fs_cross: common sampling frequency used for cross-spectra

    % Resample signals to a common frequency if necessary
    if Fs_eeg ~= Fs_emg
        [p_eeg, q_eeg] = rat(Fs_emg / Fs_eeg);
        [p_emg, q_emg] = rat(Fs_eeg / Fs_emg);

        eeg_resampled = resample(eeg_signals, p_eeg, q_eeg);
        emg_resampled = resample(emg_signals, p_emg, q_emg);

        fs_cross = Fs_emg;
    else
        eeg_resampled = eeg_signals;
        emg_resampled = emg_signals;
        fs_cross = Fs_eeg;
    end

    % Calculate the number of samples
    n_samples = size(eeg_resampled, 1);

    % Calculate the DFT length for optimal performance
    fft_length = 2^nextpow2(n_samples);  % FFT length, next power of 2 for optimal performance
    f = (0:fft_length-1) * fs_cross / fft_length;  % Frequency vector
    f = f(1:fft_length/2+1);  % Keep only the positive half of the spectrum

    S_x = zeros(length(f), size(eeg_resampled, 2));
    S_y = zeros(length(f), size(emg_resampled, 2));
    S_xy = zeros(length(f), size(eeg_resampled, 2));

    for i = 1:size(eeg_resampled, 2)
        % Calculate mean and standard deviation for normalization
        mean_eeg = mean(eeg_resampled(:, i));
        std_eeg = std(eeg_resampled(:, i));
        
        mean_emg = mean(emg_resampled(:, i));
        std_emg = std(emg_resampled(:, i));
        
        % Z-score normalization
        eeg_normalized = (eeg_resampled(:, i) - mean_eeg) / std_eeg;
        emg_normalized = (emg_resampled(:, i) - mean_emg) / std_emg;

        % Compute FFT for EEG signal
        fft_eeg = fft(eeg_normalized, fft_length);
        P2_eeg = abs(fft_eeg / n_samples);
        Peeg_filtered = P2_eeg(1:fft_length/2+1);
        Peeg_filtered(2:end-1) = 2 * Peeg_filtered(2:end-1);

        % Compute FFT for EMG signal
        fft_emg = fft(emg_normalized, fft_length);
        P2_emg = abs(fft_emg / n_samples);
        Pemg_filtered = P2_emg(1:fft_length/2+1);
        Pemg_filtered(2:end-1) = 2 * Pemg_filtered(2:end-1);

        % Compute power spectrum for EEG signal
        S_x(:, i) = Peeg_filtered;

        % Compute power spectrum for EMG signal
        S_y(:, i) = Pemg_filtered;

        % Compute cross-power spectrum between EEG and EMG signals
        cross_spec = fft_eeg .* conj(fft_emg);
        P_xy = abs(cross_spec / n_samples);
        S_xy(:, i) = P_xy(1:fft_length/2+1);
    end
end
