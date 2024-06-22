function smoothed_signal = moving_average_array(signal, window_size)
    % MOVING_AVERAGE Applica una media mobile al segnale
    
    % Controllo per il caso in cui la finestra sia troppo grande
    if window_size > length(signal)
        error('Window size cannot be larger than signal length');
    end
    
    % Costruzione del filtro per la media mobile
    b = (1/window_size) * ones(1, window_size);
    
    % Applicazione della media mobile usando conv
    smoothed_signal = conv(signal, b, 'same');
end
