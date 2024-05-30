function Hd = BandPassFilter(fs, Fcut1BPF, Fcut2BPF, BW)
%BandPassFilter Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.2 and the Signal Processing Toolbox 7.4.
% Generated on: 16-Oct-2019 09:20:14
% Modified with MATLAB R2022a 22/03/2022

% Chebyshev Type I Bandpass filter designed using FDESIGN.BANDPASS.

% All frequency values are in Hz.
% Sampling Frequency

BW = 4;
tolerance = 0.001;
Fstop1 = max([tolerance, Fcut1BPF-BW/2]);   % First Stopband Frequency
Fpass1 = Fcut1BPF;                      % First Passband Frequency
Fpass2 = Fcut2BPF;                      % Second Passband Frequency
Fstop2 = min([Fcut2BPF+BW/2, fs/2]);    % Second Stopband Frequency
Astop1 = 60;                            % First Stopband Attenuation (dB)
Apass  = 1;                             % Passband Ripple (dB)
Astop2 = 80;                            % Second Stopband Attenuation (dB)
match  = 'passband';                    % Band to match exactly

% Construct an FDESIGN object and call its CHEBY1 method.
h  = fdesign.bandpass(Fstop1, Fpass1, Fpass2, Fstop2, Astop1, Apass, ...
                      Astop2, fs);
Hd = design(h, 'cheby1', 'MatchExactly', match);

% [EOF]