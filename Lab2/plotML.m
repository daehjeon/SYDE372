function plotML(al,bl,cl, mu_al, cov_al, mu_bl, cov_bl, mu_cl, cov_cl,type)

data = [al;bl;cl];
inv_cov_al = inv(cov_al);
inv_cov_bl = inv(cov_bl);
inv_cov_cl = inv(cov_cl);
det_al = sqrt(det(cov_al));
det_bl = sqrt(det(cov_bl));
det_cl = sqrt(det(cov_cl));
minx1 = min(data);
maxx1 = max(data);
h = 20;%For parzen window
%Griding the space
x1 = minx1(1):1:maxx1(1);
x2 = minx1(2):1:maxx1(2);
[X1, X2] = meshgrid(x1, x2);


for i = 1:size(X1, 1)
    for j = 1:size(X1, 2)
    sample = [X1(i,j), X2(i,j)]';
    if type==1
        p_a = log(1/(det_al)) + (-0.5*(sample - mu_al')' * inv_cov_al * (sample - mu_al'));
        p_b = log(1/(det_bl)) + (-0.5*(sample - mu_bl')' * inv_cov_bl * (sample - mu_bl'));
        p_c = log(1/(det_cl)) + (-0.5*(sample - mu_cl')' * inv_cov_cl * (sample - mu_cl'));
    elseif type==2
        p_a = parzenF(al,sample,h);
        p_b = parzenF(bl,sample,h);
        p_c = parzenF(cl,sample,h);
    else
        disp('Error in type of estimation')
    end
    [~,ClassPrediction(i,j)] = max([p_a, p_b, p_c]);
    end
end
%plotting
% Plotting ML decision boundary in black
figure
contourf(X1, X2, ClassPrediction, 'Color', 'black');
hold on
class_c = scatter(al(:, 1), al(:, 2), 'wx');
class_d = scatter(bl(:, 1), bl(:, 2), 'bs');
class_e = scatter(cl(:, 1), cl(:, 2), 'r+');
title('ML Decision Boundary')
legend('Boundary','AL','BL','CL');
hold off
end