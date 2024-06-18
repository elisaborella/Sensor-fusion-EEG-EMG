function plotter(signal, fs, plot_title)
%PLOTTER Summary of this function goes here
%   Detailed explanation goes here
figure;
plot(signal(:,1));
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('%s - Time domain', plot_title));
legend('Filtered signal');
grid on;

% Frequency domain analysis
n = size(signal, 1);
f = (0:n-1)*(fs/n);
f = f(1:floor(n/2)+1);

% Compute FFT for the first channel
fft_filtered = fft(signal(:, 1));
P2_filtered = abs(fft_filtered / n);
P1_filtered = P2_filtered(1:floor(n/2)+1);
P1_filtered(2:end-1) = 2 * P1_filtered(2:end-1);

% Plot the frequency domain signals
figure;
plot(f, P1_filtered);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(sprintf('%s - Frequency domain', plot_title));
legend('Filtered signal');
grid on;
end

