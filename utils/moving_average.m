function smoothed_segments = moving_average(segments, window_size)
    % MOVING_AVERAGE Applica una media mobile ad ogni segmento
    %   segments: array di celle dove ogni cella contiene un segmento
    %   window_size: dimensione della finestra per la media mobile

    % Inizializza l'array di celle per i segmenti lisci
    smoothed_segments = cell(size(segments));

    % Applica la media mobile ad ogni segmento
    for i = 1:numel(segments)
        segment = segments{i};
        smoothed_segment = filter(ones(1, window_size)/window_size, 1, segment);
        smoothed_segments{i} = smoothed_segment;
    end
end