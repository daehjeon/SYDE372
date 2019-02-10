function [contour] = generateContour(mu,sigma)
% generate the standard deviation contour for given mean and sigmaq

th = linspace(0, 2*pi, 500 );
xy = [cos(th);sin(th)];
RR = chol(sigma); % cholesky decomposition
exy = xy'*RR; 
contour = [exy(:,1)+mu(:,1), exy(:,2)+mu(:,2)];
end

