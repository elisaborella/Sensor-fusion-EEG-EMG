function plot_cmc(CMC, ch)
    % PLOT_CMC Plots the Magnitude Square Coherence (CMC)
    %   CMC: Coherence matrix (frequency vs channel)
    %   ch: Channel index to plot
    %   fs: Sampling frequency (Hz)
    
    % Visualize the CMC
    figure;
    plot(CMC(:,ch));
    xlabel('Frequency (Hz)');
    ylabel('CMC');
    title(sprintf('Magnitude Square Coherence (CMC) - Channel %d', ch));
    grid on;
    hold on;
    
    % Define frequency sub-bands
    sub_bands = [6 8; 8 12; 13 20; 20 30; 13 30; 30 60; 60 80; 30 80];
    
    % Add red dashed vertical lines for each sub-band
    for j = 1:size(sub_bands, 1)
        xline(sub_bands(j, 1), 'r--');
        xline(sub_bands(j, 2), 'r--');
    end
    
    % Sub-band legend labels
    legend_labels = {'Low-α', 'α', 'Low-β', 'High-β', 'β', 'Low-γ', 'High-γ', 'γ'};
    for j = 1:size(sub_bands, 1)
        % Center the legend text between the sub-band lines
        text(mean(sub_bands(j, :)), max(CMC(:,ch))*0.9, legend_labels{j}, 'Color', 'r', 'HorizontalAlignment', 'center');
    end
    
    hold off;
end
