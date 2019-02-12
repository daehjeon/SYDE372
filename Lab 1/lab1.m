% SYDE Lab 1 - Clusters and Classification Boundaries
% Dae Jeon, ID 20377978
% Zixuan Ren, ID 20566221
% Charlotte Emily Bond, ID XXXXXXXX
% Oscar Lo, ID XXXXXXXX


%% =================================================
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

%% =================================================
% Generating Clusters
% =================================================
S_A = generateGauss(N_A, mu_A, sigma_A);
S_B = generateGauss(N_B, mu_B, sigma_B);
S_C = generateGauss(N_C, mu_C, sigma_C);
S_D = generateGauss(N_D, mu_D, sigma_D);
S_E = generateGauss(N_E, mu_E, sigma_E);

% Create Meshgrid for Classifiers 
x = min([S_A(:,1); S_B(:,1)])-1:0.05:max([S_A(:,1);S_B(:,1)])+1;
y = min([S_A(:,2);S_B(:,2)])-1:0.05:max([S_A(:,2);S_B(:,2)])+1;
[x1, y1] = meshgrid(x, y);


x = min([S_C(:,1);S_D(:,1);S_E(:,1)])-1:0.05:max([S_C(:,1);S_D(:,1);S_E(:,1)])+1;
y = min([S_C(:,2);S_D(:,2);S_E(:,2)])-1:0.05:max([S_C(:,2);S_D(:,2);S_E(:,2)])+1;
[x2, y2] = meshgrid(x, y);

% 
% % Generate scatter plot for clusters A and B
figure;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));

% Plot contour
C_A = generateContour(mu_A, sigma_A);
C_B = generateContour(mu_B, sigma_B);
hold on;
plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
hold on;
plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
title('Plot for Classes A and B (Case 1)');

% % Generate scatter plot for clusters C, D, E
figure;
scatter(S_C(:, 1), S_C(:, 2));
hold on;
scatter(S_D(:, 1), S_D(:, 2));
hold on;
scatter(S_E(:, 1), S_E(:, 2));

% Plot contour
C_C = generateContour(mu_C, sigma_C);
C_D = generateContour(mu_D, sigma_D);
C_E = generateContour(mu_E, sigma_E);
hold on;
plot(C_C(:, 1), C_C(:, 2), 'LineWidth', 3);
hold on;
plot(C_D(:, 1), C_D(:, 2), 'LineWidth', 3);
hold on;
plot(C_E(:, 1), C_E(:, 2), 'LineWidth', 3);
title('Plot for Classes C, D, and E (Case 2)');

%% =================================================
% Classifiers
% =================================================

% ///////////// CASE 1 (class A & B) //////////////
mu_true_A = mean(S_A); % true mean
mu_true_B = mean(S_B); 
sigma_true_A = cov(S_A);
sigma_true_B = cov(S_B);

% MED (Minimum Euclidean Distance)
MED_AB = generateMED_db(mu_A, mu_B, x1, y1);
MED_AB_classified = point_classifier(2, {MED_AB}, x1, y1);
figure;
plotEllipsis(sigma_A, mu_A, S_A);
plotEllipsis(sigma_B, mu_B, S_B);
contour(x1, y1, MED_AB_classified, 'LineWidth', 1);
hold on;
title('MED Decision Boundary for Class A and B');

%% GED (Generalized Euclidean Distance)

%Case A & B
distanceAB = zeros(size(x1, 1), size(y1, 2));
ged_AB = generateGED(sigma_A, mu_A, sigma_B, mu_B, x1, y1, distanceAB);
ged_AB_classified = point_classifier(2, {ged_AB}, x1, y1);
figure
title('GED for Class A and B');   
hold on
contour(x1, y1, ged_AB_classified, 'LineWidth', 1);
hold on
% scatter(S_A(:, 1), S_A(:, 2));
% hold on
% scatter(S_B(:, 1), S_B(:, 2));
legend('Decision Boundary', 'Class A', 'Class B');
plotEllipsis(sigma_A, mu_A, S_A);
plotEllipsis(sigma_B, mu_B, S_B);

%% MAP (Maximum A Posterioi)

P_A = length(S_A)/(length(S_A) + length(S_B));
P_B = length(S_B)/(length(S_A) + length(S_B));

map_AB = generate_MAP_db(mu_A, mu_B, sigma_A, sigma_B, P_A, P_B, x1, y1);
map_AB_classified = point_classifier(2, {map_AB}, x1, y1);

figure;
contour(x1, y1, map_AB_classified);
hold on;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));
hold on;
plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
hold on;
plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);

title('MAP Decision Boundary for Class A and B');

%% NN (Nearest Neighbor)
figure;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));
hold on;
generateNN_db(1, {S_A, S_B});
title("Nearest Neighbor Decision Boundary for Class A & B");

%% KNN (K-Nearest Neighbor)
figure;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));
% hold on;
% plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
% hold on;
% plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
hold on;
generateNN_db(5, {S_A, S_B});
title("K-Nearest Neighbor (k = 5) Decision Boundary for Class A & B");

%% ///////////// CASE 2 (class C, D, & E) //////////////
mu_true_C = mean(S_C); % true mean
mu_true_D = mean(S_D);
mu_true_E = mean(S_E);
sigma_true_C = cov(S_C);
sigma_true_D = cov(S_D);
sigma_true_E = cov(S_E);

%% 
% MED (Minimum Euclidean Distance)
MED_CD = generateMED_db(mu_C, mu_D, x2, y2);
MED_DE = generateMED_db(mu_D, mu_E, x2, y2);
MED_EC = generateMED_db(mu_E, mu_C, x2, y2);

% All classes
MED_CDE = point_classifier(3, {MED_CD, MED_DE, MED_EC}, x2, y2);

figure;
contour(x2, y2, MED_CDE, 'Color','Black');
hold on;

plotEllipsis(sigma_C, mu_C, S_C);
plotEllipsis(sigma_D, mu_D, S_D);
plotEllipsis(sigma_E, mu_E, S_E);

title('MED Decision Boundary for Class C, D and E');
%% 
% GED (Generalized Euclidean Distance)

GED_cd = generateGED(sigma_C, mu_C, sigma_D, mu_D, x2, y2, 0);
GED_ec = generateGED(sigma_E, mu_E, sigma_C, mu_C, x2, y2, 0);
GED_de = generateGED(sigma_D, mu_D, sigma_E, mu_E, x2, y2, 0);

%Classifying classes
GED_CDE_classified = point_classifier(3, {GED_cd, GED_de, GED_ec}, x2, y2);

%Figure
figure
title('GED for Class C, D and E');  
hold on
contour(x2,y2,GED_CDE_classified,2,'Color','Black');
hold on
legend('Decision Boundary', 'Class C', 'Class D', 'Class E');
plotEllipsis(sigma_C, mu_C, S_C);
plotEllipsis(sigma_D, mu_D, S_D);
plotEllipsis(sigma_E, mu_E, S_E);

%% 
% MAP (Maximum A Posterioi)

x = min([S_C(:,1);S_D(:,1);S_E(:,1)])-1:0.05:max([S_C(:,1);S_D(:,1);S_E(:,1)])+1;
y = min([S_C(:,2);S_D(:,2);S_E(:,2)])-1:0.05:max([S_C(:,2);S_D(:,2);S_E(:,2)])+1;
[x2, y2] = meshgrid(x, y);

P_C = length(S_C)/(length(S_C) + length(S_D) + length(S_E));
P_D = length(S_D)/(length(S_C) + length(S_D) + length(S_E));
P_E = length(S_E)/(length(S_C) + length(S_D) + length(S_E));

map_CD = generate_MAP_db(mu_C, mu_D, sigma_C, sigma_D, P_C, P_D, x2, y2);
map_DE = generate_MAP_db(mu_D, mu_E, sigma_D, sigma_E, P_D, P_E, x2, y2);
map_CE = generate_MAP_db(mu_E, mu_C, sigma_E, sigma_C, P_E, P_C, x2, y2);

map_CDE_classified = point_classifier(3, {map_CD, map_DE, map_CE}, x2, y2);

figure;
contour(x2, y2, map_CDE_classified, 'LineColor', 'b');
hold on;
scatter(S_C(:, 1), S_C(:, 2));
hold on;
scatter(S_D(:, 1), S_D(:, 2));
hold on;
scatter(S_E(:, 1), S_E(:, 2));
hold on;
plot(C_C(:, 1), C_C(:, 2), 'LineWidth', 3);
hold on;
plot(C_D(:, 1), C_D(:, 2), 'LineWidth', 3);
hold on;
plot(C_E(:, 1), C_E(:, 2), 'LineWidth', 3);
title('MAP Decision Boundary for Class C, D, and E');

%% 
% NN (Nearest Neighbor)
figure;
scatter(S_C(:, 1), S_C(:, 2));
hold on;
scatter(S_D(:, 1), S_D(:, 2));
hold on;
scatter(S_E(:, 1), S_E(:, 2));
hold on;
generateNN_db(1, {S_C, S_D, S_E});
title("Nearest Neighbor Decision Boundary for Class C, D & E");
%% 

% KNN (K-Nearest Neighbor)
figure;
scatter(S_C(:, 1), S_C(:, 2));
hold on;
scatter(S_D(:, 1), S_D(:, 2));
hold on;
scatter(S_E(:, 1), S_E(:, 2));
hold on;
generateNN_db(5, {S_C, S_D, S_E});
title("K-Nearest Neighbor (k = 5) Decision Boundary for Class C, D & E");


%% Error Analysis

% Confusion matrix of the form:
% Predicted:         A          B                      C          D         E
% Actual:      A [ a_as_a    a_as_b     ]        C [ c_as_c    c_as_d    c_as_e ]
%              B [ b_as_a    b_as_b     ]        D [ d_as_c    d_as_d    d_as_e ]
%                                                E [ e_as_c    e_as_d    e_as_e ]

%% MED error analysis

%Confusion matrix for AB
[a_as_a, a_as_b, notinuse] = MED_confusion_matrix(S_A, mu_A, mu_B, 0);
[b_as_a, b_as_b, notinuse] = MED_confusion_matrix(S_B, mu_A, mu_B, 0);

MED_confusionMatrix_AB = [a_as_a, a_as_b; b_as_a, b_as_b];

%Confusion matrix for CDE
[c_as_c, c_as_d, c_as_e] = MED_confusion_matrix(S_C, mu_C, mu_D, mu_E);
[d_as_c, d_as_d, d_as_e] = MED_confusion_matrix(S_D, mu_C, mu_D, mu_E);
[e_as_c, e_as_d, e_as_e] = MED_confusion_matrix(S_E, mu_C, mu_D, mu_E);

MED_confusionMatrix_CDE = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
 
%experimental error
% = # of wrongly classified samples / the total # of samples.
P_Error_Med_2 = (a_as_b + b_as_a)/(N_A + N_B);
P_Error_Med_3 = (c_as_d + c_as_e + d_as_c + d_as_e + e_as_c + e_as_d)/(N_C + N_D + N_E);

%% GED error analysis

GED_confusion_mtx_AB = confusion_matrix(2, {S_A, S_B}, {mu_A, mu_B}, {sigma_A, sigma_B}, 'GED');
P_Error_Ged_2 = (N_A + N_B - sum(diag(GED_confusion_mtx_AB)))/(N_A + N_B);

GED_confusion_mtx_CDE = confusion_matrix(3, {S_C, S_D, S_E}, {mu_C, mu_D, mu_E}, {sigma_C, sigma_D, sigma_E}, 'GED');
P_Error_Ged_3 = (N_C + N_D + N_E - sum(diag(GED_confusion_mtx_CDE)))/(N_C + N_D + N_E);


%% MAP error analysis

MAP_confusion_mtx_AB = confusion_matrix(2, {S_A, S_B}, {mu_A, mu_B}, {sigma_A, sigma_B}, 'MAP');

%% NN/KNN error analysis

NN_confusion_mtx_AB = confusion_matrix(2, {S_A, S_B}, {mu_A, mu_B}, {sigma_A, sigma_B}, '1NN');
P_Error_NN_2 = (N_A + N_B - sum(diag(NN_confusion_mtx_AB)))/(N_A + N_B);

NN_confusion_mtx_CDE = confusion_matrix(3, {S_C, S_D, S_E}, {mu_C, mu_D, mu_E}, {sigma_C, sigma_D, sigma_E}, '1NN');
P_Error_Ged_3 = (N_C + N_D + N_E - sum(diag(NN_confusion_mtx_CDE)))/(N_C + N_D + N_E);

KNN_confusion_mtx_AB = confusion_matrix(2, {S_A, S_B}, {mu_A, mu_B}, {sigma_A, sigma_B}, 'KNN');
P_Error_KNN_2 = (N_A + N_B - sum(diag(KNN_confusion_mtx_AB)))/(N_A + N_B);

KNN_confusion_mtx_CDE = confusion_matrix(3, {S_C, S_D, S_E}, {mu_C, mu_D, mu_E}, {sigma_C, sigma_D, sigma_E}, 'KNN');
P_Error_KNN_3 = (N_C + N_D + N_E - sum(diag(KNN_confusion_mtx_CDE)))/(N_C + N_D + N_E);
