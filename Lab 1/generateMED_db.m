function [db] = generateMED_db(mu_1,mu_2, plot_range)
%generateMED_db Generate decision boundaries based on MED
omega = (mu_1 - mu_2);
omega_0 = 0.5 * (mu_1*mu_1' - mu_2*mu_2');
x1 = linspace(plot_range(:, 1), plot_range(:, 2));
x2 = (omega_0 - x1*omega(:, 1))/omega(:, 2);
db = line (x1, x2);
end

