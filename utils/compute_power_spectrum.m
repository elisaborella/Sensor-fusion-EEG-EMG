function [S_x, S_y, S_xy, p_eeg] = compute_power_spectrum(eeg_signals, emg_signals, DUR, Fs_eeg, Fs_emg)
    % COMPUTE_POWER_SPECTRUM Calcola lo spettro di potenza di ogni canale EEG e lo spettro di potenza incrociato
    %   eeg_signals: matrice in cui ogni colonna rappresenta un canale EEG
    %   emg_signals: matrice in cui ogni colonna rappresenta un canale EMG
    %   DUR: durata di ogni segmento (in secondi)
    %   Fs_eeg: frequenza di campionamento del segnale EEG (in Hz)
    %   Fs_emg: frequenza di campionamento del segnale EMG (in Hz)
    %   S_x: matrice in cui ogni colonna rappresenta lo spettro di potenza di un canale del primo segnale
    %   S_y: matrice in cui ogni colonna rappresenta lo spettro di potenza di un canale del secondo segnale
    %   S_xy: matrice in cui ogni colonna rappresenta lo spettro di potenza incrociato tra canali corrispondenti

    % Parametri
    [n_samples_eeg, n_channels_eeg] = size(eeg_signals); % numero di campioni e canali EEG
    [n_samples_emg, n_channels_emg] = size(emg_signals); % numero di campioni e canali EMG

    n_samples = min(n_samples_emg, n_samples_eeg);
    n_channels = min(n_channels_emg, n_channels_eeg);

    % Uniformare la lunghezza dei segnali tramite resampling
    [p_eeg, q_eeg] = rat(Fs_emg / Fs_eeg);
    eeg_resampled = resample(eeg_signals, p_eeg, q_eeg);

    % Ridimensionare EMG per corrispondere a EEG resampled
    [p_emg, q_emg] = rat(Fs_eeg / Fs_emg);
    emg_resampled = resample(emg_signals, p_emg, q_emg);

    % Determinare il numero di segmenti per l'analisi della durata specificata
    segment_length = DUR * Fs_eeg; % numero di campioni per segmento
    n_segments = floor(n_samples / segment_length); % numero di segmenti completi

    % Inizializzare matrici per i risultati
    S_x = zeros(segment_length/2+1, n_channels, n_segments);
    S_y = zeros(segment_length/2+1, n_channels, n_segments);
    S_xy = zeros(segment_length/2+1, n_channels, n_segments);

    % Loop su ogni segmento
    for seg = 1:n_segments
        for i = 1:n_channels
            % Segmentare il segnale EEG
            segment_eeg = eeg_resampled((seg-1)*segment_length+1:seg*segment_length, i);
            % Calcolare la DFT e lo spettro di potenza
            eeg_dft = fft(segment_eeg);
            s_x = abs(eeg_dft / segment_length).^2;
            S_x(:, i, seg) = s_x(1:segment_length/2+1);
       
            % Segmentare il segnale EMG
            segment_emg = emg_resampled((seg-1)*segment_length+1:seg*segment_length, i);
            % Calcolare la DFT e lo spettro di potenza
            emg_dft = fft(segment_emg);
            s_y = abs(emg_dft / segment_length).^2;
            S_y(:, i, seg) = s_y(1:segment_length/2+1);
        
            % Calcolare lo spettro di potenza incrociato tra i canali corrispondenti
            segment_eeg = eeg_resampled((seg-1)*segment_length+1:seg*segment_length, i);
            segment_emg = emg_resampled((seg-1)*segment_length+1:seg*segment_length, i);
            
            eeg_dft = fft(segment_eeg);
            emg_dft = fft(segment_emg);
            
            s_xy = eeg_dft .* conj(emg_dft) / segment_length;
            S_xy(:, i, seg) = abs(s_xy(1:segment_length/2+1));
        end
    end
end
