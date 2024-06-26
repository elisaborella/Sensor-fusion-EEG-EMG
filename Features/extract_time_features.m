

function [features_time] = extract_time_features(x, fs)
% -- Time domain uni-dimensional features --
%
% 1: MIN:  min amplitude value
% 2: MAX:  max amplitude value
% 3: MEAN: mean amplitude value
% 4: MED:  median amplitude value
% 5: SD:   standard deviation from the mean
% 6: VAR:  variance from the mean
% 7: PP:   peak-to-peak distance (range)
% 8: ZC:   zero-crossings
% 9: AUC:  area under curve
% 10: RMS:  root mean square (entire segment)
% 11: MP:   mean (amplitude) power
% 12: MAV:  mean absolute value
% 13: EN:  signal's energy. Could be computed also in the frequency domain (area under the power spectrum curve). See Parseval's theorem.
% 14: WL:   waveform length
% 15: SK:   skewness
% 16: KUR:  kurtosis


%% Initialization
features_time = [];


%% Convert input data to single or double
x = double(x); % Convert to double data type

%% Time
t = 1/fs:1/fs:length(x)/fs; %[s]


% Minimum
MIN = min(x);

% Maximum
MAX = max(x);

% Mean amplitude
MEAN = mean(x);

% Median amplitude
MED = median(x);

% Stand. deviation
SD = std(x);

% Variance
VAR = var(x);

% Peak-to-peak distance (range)
PP = range(x); %max-min

% ZC:   zero-crossings (see check_ZC)
[MYZC, myZCt, myZCx, xs] = myZC(t,x);
if ~isnan(MYZC)
    ZC = length(MYZC);
else
    ZC = nan;
end


% AUC:  area under curve (see check_AUC)
myInt = cumtrapz(t,abs(x));
myIntv = @(a,b) max(myInt(t<=b)) - min(myInt(t>=a));
AUC = myIntv(t(1), t(end));


% Root mean square (entire segment)
RMS = rms(x);


% Mean (amplitude) power
MP = norm(x)^2/length(x);


% MAV:  mean absolute value
MAV = mean(abs(x));


% EN:  signal's energy. Could be computed also in the frequency domain (area under the power spectrum curve). See Parseval's theorem.
EN = norm(x)^2;


% WL:   waveform length (see:
% https://www.researchgate.net/publication/317056066_Counting_Grasping_Action_Using_Force_Myography_An_Exploratory_Study_With_Healthy_Individuals/figures?lo=1)
WL = sum(diff(x));


% Skewness
SK = skewness(x);

% Kurtosis
KUR = kurtosis(x);



%% Feature vector
features_time = [MIN, MAX, MEAN, MED, SD, VAR, PP, ZC, AUC, RMS, MP, MAV, EN, WL, SK, KUR];

end