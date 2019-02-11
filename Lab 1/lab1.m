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
figure;
generateMED_db(mu_true_A, mu_true_B, [-5, 20]);
hold on;
scatter(S_A(:, 1), S_A(:, 2));
hold on;
scatter(S_B(:, 1), S_B(:, 2));
hold on;
plot(C_A(:, 1), C_A(:, 2), 'LineWidth', 3);
hold on;
plot(C_B(:, 1), C_B(:, 2), 'LineWidth', 3);
title('MED Decision Boundary for Class A and B');

%% GED (Generalized Euclidean Distance)
% Create Meshgrid for Classifiers 
x = min([S_A(:,1); S_B(:,1)])-1:0.05:max([S_A(:,1);S_B(:,1)])+1;
y = min([S_A(:,2);S_B(:,2)])-1:0.05:max([S_A(:,2);S_B(:,2)])+1;
[x1, y1] = meshgrid(x, y);

%Case A & B
distanceAB = zeros(size(x1, 1), size(y1, 2));
ged_AB = generateGED(sigma_A, mu_A, sigma_B, mu_B, x1, y1, distanceAB);
figure
title('GED for Class A and B');   
hold on
contour(x1, y1, ged_AB, [0, 0], 'LineWidth', 1);
hold on
% scatter(S_A(:, 1), S_A(:, 2));
% hold on
% scatter(S_B(:, 1), S_B(:, 2));
legend('Decision Boundary', 'Class A', 'Class B');
plotEllipsis(sigma_A, mu_A, S_A);
plotEllipsis(sigma_B, mu_B, S_B);

%% MAP (Maximum A Posterioi)


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
figure;
generateMED_db(mu_true_C, mu_true_D, [9.8, 10.2]);
hold on;
generateMED_db(mu_true_C, mu_true_E, [-5, 20]);
hold on;
generateMED_db(mu_true_D, mu_true_E, [-5, 20]);
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
title('MED Decision Boundary for Class C, D, and E');
%% 
% GED (Generalized Euclidean Distance)

x = min([S_C(:,1);S_D(:,1);S_E(:,1)])-1:0.05:max([S_C(:,1);S_D(:,1);S_E(:,1)])+1;
y = min([S_C(:,2);S_D(:,2);S_E(:,2)])-1:0.05:max([S_C(:,2);S_D(:,2);S_E(:,2)])+1;
[x2, y2] = meshgrid(x, y);

GED_cd = generateGED(sigma_C, mu_C, sigma_D, mu_D, x2, y2, 0);
GED_ec = generateGED(sigma_E, mu_E, sigma_C, mu_C, x2, y2, 0);
GED_de = generateGED(sigma_D, mu_D, sigma_E, mu_E, x2, y2, 0);

%Classifying classes
GED2 = zeros(size(x2, 1), size(y2, 2));
for i=1:size(x2, 1)
    for j=1:size(y2, 2)
        c = 1; d = 2; e = 3;
        if (GED_cd(i,j) < 0 && GED_ec(i,j) > 0)
            % If distance is less than 0 for cd and greater than 0 for ec, it is part of class C
            GED2(i, j) = c;
        elseif (GED_cd(i,j) > 0 && GED_de(i,j) < 0)
            % If distance is greater than 0 for cd and greater than 0 for de, it is part of class D
            GED2(i, j) = d;
        elseif (GED_de(i,j) > 0 && GED_ec(i,j) < 0)
            % If distance is greater than 0 for de and less than 0 for ec, it is part of class E
            GED2(i, j) = e;
        else
            disp('something is wrong');
        end
    end
end

%Figure
figure
title('GED for Class C, D and E');  
hold on
contour(x2,y2,GED2,2,'Color','Black');
hold on
legend('Decision Boundary', 'Class C', 'Class D', 'Class E');
plotEllipsis(sigma_C, mu_C, S_C);
plotEllipsis(sigma_D, mu_D, S_D);
plotEllipsis(sigma_E, mu_E, S_E);

%% 
% MAP (Maximum A Posterioi)


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