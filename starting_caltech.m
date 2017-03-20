function fu = starting_caltech()

close all; clear variables; clc;
load words-model.mat

nsamples = 60; % 1000 is the maximum
descriptor_size = 12000;
samples_fig = figure;

% Load 'butterfly' examples
butterfly_dir = '101_ObjectCategories/butterfly/';
butterfly_images=dir([butterfly_dir '*.jpg']);

butterfly_samples = zeros(descriptor_size,nsamples);
for i=1:nsamples
    im=imread([butterfly_dir butterfly_images(i).name]);
    hist = getImageDescriptor(model, im);
    butterfly_samples(:,i) = hist;
    disp(i)
    % samples_fig; imshow(uint8(t1')); pause(0.1);
end

% Load 'dragonfly' examples
dragonfly_dir = '101_ObjectCategories/grand_piano/';
dragonfly_images=dir([dragonfly_dir '*.jpg']);

dragonfly_samples = zeros(descriptor_size,nsamples);
for i=1:nsamples
    im=imread([dragonfly_dir dragonfly_images(i).name]);
    hist = getImageDescriptor(model, im);
    dragonfly_samples(:,i) = hist;
    disp(i)
    % samples_fig; imshow(uint8(t1')); pause(0.1);
end

% BUILD DISTANCE MATRIX W AND LABEL MATRIX fl, AS DONE PREVIOUSLY WITH THE
% MNIST DATASET

% 60 samples => Try 10 tagged + 50 untagged
% Tagged samples
samples(:, 1:10) = butterfly_samples(:, 1:10);
samples(:, 11:20) = dragonfly_samples(:, 1:10);
% Untagged samples
samples(:, 21:70) = butterfly_samples(:, 11:60);
samples(:, 71:120) = dragonfly_samples(:, 11:60);

% Calculate weights matrix
sigma_d = 0.01;
w = zeros(nsamples*2, nsamples*2);
for a = 1:i*2
    for b = 1:i*2
        w(a, b) = exp(-sum(((samples(:, a) - samples(:, b)).^2)./(sigma_d^2)));
    end    
end

% Tagged samples
fl = zeros(20, 2);
fl(1:10, 1) = 1;
fl(11:20, 2) = 1;

[fu, fu_CMN] = harmonic_function(w, fl);


% -------------------------------------------------------------------------
function im = standarizeImage(im)
% -------------------------------------------------------------------------

im = im2single(im) ;
if size(im,1) > 480, im = imresize(im, [480 NaN]) ; end

% -------------------------------------------------------------------------
function hist = getImageDescriptor(model, im)
% -------------------------------------------------------------------------

im = standarizeImage(im) ;
width = size(im,2) ;
height = size(im,1) ;
numWords = size(model.vocab, 2) ;

% get PHOW features
[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

% quantize local descriptors into visual words
switch model.quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 50)) ;
end

for i = 1:length(model.numSpatialX)
  binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
  binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

  % combined quantization
  bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                 binsy,binsx,binsa) ;
  hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
  hist = vl_binsum(hist, ones(size(bins)), bins) ;
  hists{i} = single(hist / sum(hist)) ;
end
hist = cat(1,hists{:}) ;
hist = hist / sum(hist) ;