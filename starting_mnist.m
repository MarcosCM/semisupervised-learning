close all; clear variables; clc;

nsamples = 1000; % 1000 is the maximum
im_size = 28;
samples_fig = figure;

% Load '0' examples
fid0=fopen('mnist/data0','r');

zero_samples = zeros(im_size*im_size,nsamples);
for i=1:nsamples
    [t1,N]=fread(fid0,[im_size im_size],'uchar');
    zero_samples(:,i) = reshape(uint8(t1'),im_size*im_size,1);
    % samples_fig; imshow(uint8(t1')); pause(0.1);
end

fclose(fid0);

% Load '8' examples
fid1=fopen('mnist/data8','r');

eight_samples = zeros(im_size*im_size,nsamples);
for i=1:nsamples
    [t1,N]=fread(fid1,[im_size im_size],'uchar');
    eight_samples(:,i) = reshape(uint8(t1'),im_size*im_size,1);
    % samples_fig; imshow(uint8(t1')); pause(0.1);
end

fclose(fid1);

% eight_samples = zeros(im_size*im_size,nsamples);

% BUILD DISTANCE MATRIX W CONTAINING THE DISTANCE FROM EVERY SAMPLE TO
% EVERY OTHER ONE. THE FIRST ENTRIES ARE FOR THE LABELED DATA

% 1000 samples => Try 100 tagged + 900 untagged
% Tagged samples
samples(:, 1:100) = eight_samples(:, 1:100);
samples(:, 101:200) = zero_samples(:, 1:100);
% Untagged samples
samples(:, 201:1100) = eight_samples(:, 101:1000);
samples(:, 1101:2000) = zero_samples(:, 101:1000);

% Calculate weights matrix
sigma_d = 380;
w = zeros(nsamples*2, nsamples*2);
for a = 1:i*2
    for b = 1:i*2
        w(a, b) = exp(-sum(((samples(:, a) - samples(:, b)).^2)./(sigma_d^2)));
    end    
end

% BUILD LABEL MATRIX fl CONTAINING THE LABELS FOR EVERY LABEL SAMPLE.
% FORMAT: 2 COLUMNS, ONE PER CLASS; L ROWS, ONE PER SAMPLE. 1 IF THE SAMPLE
% BELONGS TO THE CLASS, ZERO IF IT DOES NOT

% Tagged samples
fl = zeros(200, 2);
fl(1:100, 1) = 1;
fl(101:200, 2) = 1;

[fu, fu_CMN] = harmonic_function(w, fl);