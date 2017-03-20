function [ samples ] = label_samples( samples, nlabels )

for i = 1:nlabels
    % Number of columns = number of samples
    idx = randi(size(samples,2));
    aux = samples(idx);
    samples(idx) = samples(i);
    samples(i) = aux;
end

end