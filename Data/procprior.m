%% compute mean and covaraince
function [mean_w, sigma_w, w] = procprior( filename, outfile )
% w = importdata( filename );
% ndims = size(w, 2);

fin = fopen(filename, 'r');
nsamples = fread(fin, 1, 'int32');
ndims = fread(fin, 1, 'int32');
w = fread(fin, nsamples*ndims, 'single');
size(w)
w = reshape(w, ndims, nsamples );
w = w';
fclose(fin);



mean_w = mean(w);
sigma_w = cov(w);

% fix the cov matrix
sigma_w(1, 1) = trace(sigma_w)/(ndims-1);

fid = fopen(outfile, 'w');
fwrite(fid, ndims, 'int');
fwrite(fid, w(1,:), 'single');  % 0th id weight / neutral expression weight
fwrite(fid, mean_w, 'single');
fwrite(fid, sigma_w, 'single');
fclose(fid);

end
