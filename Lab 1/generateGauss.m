function [cluster] = generateGauss(N, mu, sigma)
% generate the cluster for given sample size, mean, and sigma

R = chol(sigma);
cluster = repmat(mu, N, 1) + randn(N, 2)*R;
end

