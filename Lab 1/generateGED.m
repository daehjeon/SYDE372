function [S_trans, mu_trans] = generateGED(sigma, mu, S)
%generateGED Transform to generalized Euclidean metric
diag_cov = diag(eig(sigma)); 
[V, phi] = eig(sigma);
W = diag_cov ^ (-0.5) * phi';
mu_trans = W * mu';
S_trans = S * W;
end

