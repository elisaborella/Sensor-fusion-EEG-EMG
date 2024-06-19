function plot_power_spectrum(S_x, S_y, S_xy, Fs)
    % PLOT_POWER_SPECTRUM Plots the average power spectra and cross-power spectrum
    %   S_x: Power spectrum of EEG signals, size [frequency_bins, n_channels]
    %   S_y: Power spectrum of EMG signals, size [frequency_bins, n_channels]
    %   S_xy: Cross-power spectrum between EEG and EMG signals, size [frequency_bins, n_channels]
    %   Fs: Sampling frequency of the signals (in Hz)

    % Determine frequency axis
    n_channels = size(S_x, 2);
    frequency_bins = size(S_x, 1);
    f = (0:(frequency_bins-1)) * (Fs / frequency_bins);

    for ch = 1:n_channels
        S_x_avg = mean(S_x(:, ch, :), 3);
        S_y_avg = mean(S_y(:, ch, :), 3);
        S_xy_avg = mean(S_xy(:, ch, :), 3);
        
        % Plot EEG power spectrum
        figure;
        subplot(3, 1, 1);
        plot(f, abs(S_x_avg));
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('EEG Power Spectrum - Channel %d', ch));
        grid on;

        % Plot EMG power spectrum
        subplot(3, 1, 2);
        plot(f, abs(S_y_avg));
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('EMG Power Spectrum - Channel %d', ch));
        grid on;

        % Plot cross-power spectrum (EEG-EMG)
        subplot(3, 1, 3);
        plot(f, abs(S_xy_avg));
        xlabel('Frequency (Hz)');
        ylabel('Cross Power (dB)');
        title(sprintf('Cross Power Spectrum (EEG-EMG) - Channel %d', ch));
        grid on;
    end
end
