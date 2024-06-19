function [segments, t, stride_samples, window_length_samples] = segmentation(signal, fs, window_length_ms, overlap_percentage)
    % SEGMENTATION Segmenta il segnale in finestre sovrapposte
    %   signal: matrice in cui ogni colonna rappresenta un canale
    %   fs: frequenza di campionamento del segnale (in Hz)
    %   window_length_ms: lunghezza della finestra in millisecondi
    %   overlap_percentage: percentuale di sovrapposizione tra le finestre

    % Calcolare la lunghezza della finestra in campioni
    window_length_samples = round(window_length_ms / 1000 * fs);
    % Calcolare lo stride (numero di campioni tra l'inizio di una finestra e l'inizio della successiva)
    stride_samples = round(window_length_samples * (1 - overlap_percentage / 100));

    t = (0:window_length_samples-1) / fs;

    % Inizializzare l'array di celle per contenere i segmenti
    segments = {};

    % Ottenere il numero di campioni e canali del segnale
    num_samples = size(signal, 1);

    % Segmentare il segnale
    start_idx = 1;
    while start_idx + window_length_samples - 1 <= num_samples
        end_idx = start_idx + window_length_samples - 1;
        segment = signal(start_idx:end_idx, :);
        segments{end + 1} = segment;
        start_idx = start_idx + stride_samples;
    end
end
