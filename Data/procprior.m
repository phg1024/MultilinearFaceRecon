%% compute mean and covaraince
function [mean_w, sigma_w, w] = procprior( filename, outfile )
%w = importdata( filename );
fin = fopen(filename, 'r');
nsamples = fread(fin, 1, 'int32');
ndims = fread(fin, 1, 'int32');
w = fread(fin, nsamples*ndims, 'single');
size(w)
w = reshape(w, nsamples, ndims);
fclose(fin);

mean_w = mean(w);
sigma_w = cov(w);

fid = fopen(outfile, 'w');
fwrite(fid, ndims, 'int');
fwrite(fid, mean_w, 'single');
fwrite(fid, sigma_w, 'single');
fclose(fid);

end
