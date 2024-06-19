function [S_x, S_y, S_xy, f] = compute_power_spectrum(eeg_signals, emg_signals, Fs_eeg, Fs_emg)
    % COMPUTE_POWER_SPECTRUM Calcola lo spettro di potenza di ogni canale EEG e lo spettro di potenza incrociato
    %   eeg_signals: matrice in cui ogni colonna rappresenta un canale EEG
    %   emg_signals: matrice in cui ogni colonna rappresenta un canale EMG
    %   Fs_eeg: frequenza di campionamento del segnale EEG (in Hz)
    %   Fs_emg: frequenza di campionamento del segnale EMG (in Hz)
    %   S_x: matrice in cui ogni colonna rappresenta lo spettro di potenza di un canale del primo segnale
    %   S_y: matrice in cui ogni colonna rappresenta lo spettro di potenza di un canale del secondo segnale
    %   S_xy: matrice in cui ogni colonna rappresenta lo spettro di potenza incrociato tra canali corrispondenti
    %   f: vettore delle frequenze

    % Uniformare la lunghezza dei segnali tramite resampling se necessario
    if Fs_eeg ~= Fs_emg
        [p_eeg, q_eeg] = rat(Fs_emg / Fs_eeg);
        [p_emg, q_emg] = rat(Fs_eeg / Fs_emg);

        eeg_resampled = resample(eeg_signals, p_eeg, q_eeg);
        emg_resampled = resample(emg_signals, p_emg, q_emg);
    else
        eeg_resampled = eeg_signals;
        emg_resampled = emg_signals;
    end

    % Calcolare il numero di campioni
    n_samples = size(eeg_resampled, 1);

    % Calcolare la DFT per ottenere lo spettro di potenza
    fft_length = 2^nextpow2(n_samples);  % Lunghezza della FFT, potenza di 2 per prestazioni ottimali
    f = (0:fft_length-1) * Fs_eeg / fft_length;  % Vettore delle frequenze
    f = f(1:fft_length/2+1);  % Conserva solo metà dello spettro (parte positiva)

    S_x = zeros(length(f), size(eeg_resampled, 2));
    S_y = zeros(length(f), size(emg_resampled, 2));
    S_xy = zeros(length(f), size(eeg_resampled, 2));

    for i = 1:size(eeg_resampled, 2)
        % Calcolo della media e della deviazione standard
        mean_eeg = mean(eeg_resampled(:,i));
        std_eeg = std(eeg_resampled(:,i));
        
        mean_emg = mean(emg_resampled(:,i));
        std_emg = std(emg_resampled(:,i));
        
        % Normalizzazione Z-score
        eeg_normalized = (eeg_resampled(:,i) - mean_eeg) / std_eeg;
        emg_normalized = (emg_resampled(:,i) - mean_emg) / std_emg;

        % Calcolare la FFT per il segnale EEG
        fft_eeg = fft(eeg_normalized, fft_length);
        P2_eeg = abs(fft_eeg / n_samples);
        Peeg_filtered = P2_eeg(1:fft_length/2+1);
        Peeg_filtered(2:end-1) = 2 * Peeg_filtered(2:end-1);

        % Calcolare la FFT per il segnale EMG
        fft_emg = fft(emg_normalized, fft_length);
        P2_emg = abs(fft_emg / n_samples);
        Pemg_filtered = P2_emg(1:fft_length/2+1);
        Pemg_filtered(2:end-1) = 2 * Pemg_filtered(2:end-1);

        % Calcolare lo spettro di potenza per il segnale EEG
        S_x(:, i) = Peeg_filtered;

        % Calcolare lo spettro di potenza per il segnale EMG
        S_y(:, i) = Pemg_filtered;

        % Calcolare lo spettro di potenza incrociato tra EEG ed EMG
        cross_spec = fft_eeg .* conj(fft_emg);
        P_xy = abs(cross_spec / n_samples);
        S_xy(:, i) = P_xy(1:fft_length/2+1);
    end
end
