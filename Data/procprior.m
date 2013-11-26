%% compute mean and covaraince
function [mean_w, sigma_w] = proprior( filename, outfile )
w = importdata( filename );

[nsamples, ndims] = size(w);

mean_w = mean(w);
sigma_w = cov(w);

fid = fopen(outfile, 'w');
fwrite(fid, ndims, 'int');
fwrite(fid, mean_w, 'single');
fwrite(fid, sigma_w, 'single');
fclose(fid);

end
