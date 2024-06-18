% Funzione per plottare gli spettri di potenza
function plot_power_spectrum(S_x, S_y, S_xy, Fs, DUR)
    n_channels = size(S_x, 2);
    n_segments = size(S_x, 3);
    segment_length = DUR * Fs;
    f = (0:(segment_length/2)) * (Fs / segment_length);
    for ch = 1:n_channels
        S_x_avg = mean(S_x(:, ch, :), 3);
        S_y_avg = mean(S_y(:, ch, :), 3);
        S_xy_avg = mean(S_xy(:, ch, :), 3);
        figure;
        subplot(3, 1, 1);
        plot(f, 10*log10(S_x_avg));
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('EEG Power Spectrum - Channel %d', ch));
        grid on;
        subplot(3, 1, 2);
        plot(f, 10*log10(S_y_avg));
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title(sprintf('EMG Power Spectrum - Channel %d', ch));
        grid on;
        subplot(3, 1, 3);
        plot(f, 10*log10(S_xy_avg));
        xlabel('Frequency (Hz)');
        ylabel('Cross Power (dB)');
        title(sprintf('Cross Power Spectrum (EEG-EMG) - Channel %d', ch));
        grid on;
    end
end