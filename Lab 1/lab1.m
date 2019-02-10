% Clusters and Classifcation Boundaries

% =================================================
% Class data
% =================================================

% Class A
N_A = 200;
mu_A = [5 10];
sigma_A = [8 0; 0 4];

% Class B
N_B = 200;
mu_B = [10 15];
sigma_B = [8 0; 0 4];

% Class C
N_C = 100;
mu_C = [5 10];
sigma_C = [8 4; 4 40];

% Class D
N_D = 200;
mu_D = [15 10];
sigma_D = [8 0; 0 8];

% Class E
N_E = 150;
mu_E = [10 5];
sigma_E = [10 -5; -5 20];

% =================================================
% Generating Clusters
% =================================================
% S_A = generateGauss(N_A, mu_A, sigma_A);
% S_B = generateGauss(N_B, mu_B, sigma_B);
% S_C = generateGauss(N_C, mu_C, sigma_C);
% S_D = generateGauss(N_D, mu_D, sigma_D);
% S_E = generateGauss(N_E, mu_E, sigma_E);
% 
% % Generate scatter plot for clusters A and B
% figure;
% scatter(S_A(:, 1), S_A(:, 2));
% hold on;
% scatter(S_B(:, 1), S_B(:, 2));
% 
% % Plot contour
% C_A = generateContour(mu_A, sigma_A);
% C_B = generateContour(mu_B, sigma_B);
% hold on;
% plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
% title('Plot for Classes A and B (Case 1)');
% 
% % Generate scatter plot for clusters C, D, E
% figure;
% scatter(S_C(:, 1), S_C(:, 2));
% hold on;
% scatter(S_D(:, 1), S_D(:, 2));
% hold on;
% scatter(S_E(:, 1), S_E(:, 2));
% 
% % Plot contour
% C_C = generateContour(mu_C, sigma_C);
% C_D = generateContour(mu_D, sigma_D);
% C_E = generateContour(mu_E, sigma_E);
% hold on;
% plot(C_C(:, 1), C_C(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_D(:, 1), C_D(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_E(:, 1), C_E(:, 2), 'LineWidth', 3);
% title('Plot for Classes C, D, and E (Case 2)');

% =================================================
% Classifiers
% =================================================

% ///////////// CASE 1 (class A & B) //////////////
mu_true_A = mean(S_A); % true mean
mu_true_B = mean(S_B); 
sigma_true_A = cov(S_A);
sigma_true_B = cov(S_B);

% MED (Minimum Euclidean Distance)
% figure;
% generateMED_db(mu_true_A, mu_true_B, [-5, 20]);
% hold on;
% scatter(S_A(:, 1), S_A(:, 2));
% hold on;
% scatter(S_B(:, 1), S_B(:, 2));
% hold on;
% plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
% title('MED Decision Boundary for Class A and B');

% GED (Generalized Euclidean Distance) - TODO: Need to be normalized???
% [S_A_trans, mu_A_trans] = generateGED(sigma_A, mu_A, S_A);
% C_A_ged = generateContour(mu_A_trans', eye(2)); % Using identity matrix (eye) to produce unit contour???
% 
% figure;
% scatter(S_A_trans(:, 1), S_A_trans(:, 2));
% hold on;
% plot(C_A_ged(:, 1), C_A_ged(:, 2), 'LineWidth', 3);

% MAP (Maximum A Posterioi)

% NN (Nearest Neighbor)
NN_db_ab = generateNN_db(S_B, S_A);
figure;
line(NN_db_ab(:, 1), NN_db_ab(:, 2), 'LineWidth', 2);
hold on;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));
hold on;
plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
hold on;
plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
hold on;
generateMED_db(mu_true_A, mu_true_B, [-5, 20]);
title("Nearest Neighbor Decision Boundary for Class A & B");

% KNN (K-Nearest Neighbor)



% ///////////// CASE 2 (class C, D, & E) //////////////
mu_true_C = mean(S_C); % true mean
mu_true_D = mean(S_D);
mu_true_E = mean(S_E);
sigma_true_C = cov(S_C);
sigma_true_D = cov(S_D);
sigma_true_E = cov(S_E);

% MED (Minimum Euclidean Distance)
% figure;
% generateMED_db(mu_true_C, mu_true_D, [9.8, 10.2]);
% hold on;
% generateMED_db(mu_true_C, mu_true_E, [-5, 20]);
% hold on;
% generateMED_db(mu_true_D, mu_true_E, [-5, 20]);
% hold on;
% scatter(S_C(:, 1), S_C(:, 2));
% hold on;
% scatter(S_D(:, 1), S_D(:, 2));
% hold on;
% scatter(S_E(:, 1), S_E(:, 2));
% hold on;
% plot(C_C(:, 1), C_C(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_D(:, 1), C_D(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_E(:, 1), C_E(:, 2), 'LineWidth', 3);
% title('MED Decision Boundary for Class C, D, and E');

% GED (Generalized Euclidean Distance) - TODO: Need to be normalized???

% MAP (Maximum A Posterioi)

% NN (Nearest Neighbor)

% KNN (K-Nearest Neighbor)
