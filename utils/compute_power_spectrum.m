function [S_x, S_y, S_xy] = compute_power_spectrum(eeg_signals, emg_signals, DUR, Fs_eeg, Fs_emg)
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
    
    % Verifica che abbiano lo stesso numero di canali
    if n_channels_eeg ~= n_channels_emg
        error('I segnali devono avere lo stesso numero di canali.');
    end
    
    n_channels = n_channels_eeg;
    
    ts_eeg = n_samples_eeg * Fs_eeg;
    ts_emg = n_samples_emg * Fs_emg;
    
    % Numero di segmenti
    n_segments_eeg = floor(ts_eeg / DUR);
    n_segments_emg = floor(ts_emg / DUR);
    
    % Numero minimo di segmenti tra i due segnali
    n_segments = min(n_segments_eeg, n_segments_emg);
    
    % Prealloca le matrici per gli spettri di potenza e lo spettro di potenza incrociato
    S_x = zeros(floor(n_segments / 2) + 1, n_channels);
    S_y = zeros(floor(n_segments / 2) + 1, n_channels);
    S_xy = zeros(floor(n_segments / 2) + 1, n_channels);

    % Calcola la potenza spettrale per ogni segmento e canale
    for segment = 1:n_segments
        for channel = 1:n_channels
            % Estrai il segmento per il canale corrente
            eeg_segment = eeg_signals((segment-1)*DUR + (1:DUR), channel);
            emg_segment = emg_signals((segment-1)*DUR + (1:DUR), channel);
            
            % Calcola la FFT per entrambi i segnali
            fft_result_eeg = fft(eeg_segment);
            fft_result_emg = fft(emg_segment);
            
            % Calcola lo spettro di potenza per il primo segnale
            P2_eeg = abs(fft_result_eeg / DUR);
            P1_eeg = P2_eeg(1:floor(DUR / 2) + 1);
            P1_eeg(2:end-1) = 2 * P1_eeg(2:end-1);
            
            % Calcola lo spettro di potenza per il secondo segnale
            P2_emg = abs(fft_result_emg / DUR);
            P1_emg = P2_emg(1:floor(DUR / 2) + 1);
            P1_emg(2:end-1) = 2 * P1_emg(2:end-1);
            
            % Calcola lo spettro di potenza incrociato
            cross_P2 = fft_result_eeg(1:length(P1_eeg)) .* conj(fft_result_emg(1:length(P1_eeg))) / sqrt(DUR * DUR);
            cross_P1 = abs(cross_P2(1:floor(DUR / 2) + 1));
            cross_P1(2:end-1) = 2 * cross_P1(2:end-1);
            
            % Accumula gli spettri di potenza e incrociati nelle rispettive matrici
            S_x(:, channel) = S_x(:, channel) + P1_eeg;
            S_y(:, channel) = S_y(:, channel) + P1_emg;
            S_xy(:, channel) = S_xy(:, channel) + cross_P1;
        end
    end
    
    % Media gli spettri di potenza e incrociati sui segmenti
    S_x = S_x / n_segments;
    S_y = S_y / n_segments;
    S_xy = S_xy / n_segments;
end
