load dataset

fs = 2000;

[features, parameters] = universal_feature_extraction(dataset, fs, 'emg');

time = 0:1/fs:6000/fs-1/fs;

figure, plot(time, squeeze(dataset(:,1,2)),'k'), grid, xlabel('Time [s]'), ylabel('Amplitude [mV]'), title('Exemplary EMG signal')

parameters.featnames

RMS_values = squeeze(features(10,1,:));