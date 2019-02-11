function grid_plot = generate_MAP_db(mu1, mu2, sigma1,sigma2, P1, P2, x, y)
%generate_MAP_db Generates the MAP decision boundary

sigma_1_inv = inv(sigma1);
sigma_2_inv = inv(sigma2);

Q_0 = sigma_1_inv - sigma_2_inv;
Q_1 = 2*(sigma2\mu2' - sigma1\mu1');
Q_2 = mu1 * sigma_1_inv * mu1' - mu2 * sigma_2_inv * mu2';
Q_3 = log(P2 / P1);
Q_4 = log(det(sigma1) / det(sigma2));

grid_plot = zeros(size(x, 1), size(y, 2));

for i = 1:size(x, 1)
    for j = 1:size(y, 2)
        X = [x(i, j), y(i, j)];
        grid_plot(i, j) = X*Q_0*X' + X*Q_1 + Q_2 + 2*Q_3 + Q_4;
    end
end
end

