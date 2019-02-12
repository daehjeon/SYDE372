function mtx = confusion_matrix(size, true_stat, means, covariance, method)
%confusion_matrix Returns the confusion matrix from the results given

if (size == 2) 
    sample_A = true_stat{1};
    sample_B = true_stat{2};
    
    if (method == 'GED')
        [a_as_a, a_as_b, tmp] = getGEDConfusionMtx(means, covariance, sample_A, 2);
        [b_as_a, b_as_b, tmp] = getGEDConfusionMtx(means, covariance, sample_B, 2);
        mtx = [a_as_a, a_as_b; b_as_a, b_as_b];
    elseif (method == '1NN')
        [a_as_a, a_as_b, tmp] = getNNConfusionMtx(1, 1, {sample_A, sample_B}, 2);
        [b_as_a, b_as_b, tmp] = getNNConfusionMtx(1, 2, {sample_A, sample_B}, 2);
        mtx = [a_as_a, a_as_b; b_as_a, b_as_b];
    elseif (method == 'MAP')
        mtx = getMAPConfusionMtx(means, covariance, true_stat);
    elseif (method == 'KNN')
        [a_as_a, a_as_b, tmp] = getNNConfusionMtx(5, 1, {sample_A, sample_B}, 2);
        [b_as_a, b_as_b, tmp] = getNNConfusionMtx(5, 2, {sample_A, sample_B}, 2);
        mtx = [a_as_a, a_as_b; b_as_a, b_as_b];
    end
    
elseif (size == 3)
    sample_C = true_stat{1};
    sample_D = true_stat{2};
    sample_E = true_stat{3};

    if (method == 'GED')
        [c_as_c, c_as_d, c_as_e] = getGEDConfusionMtx(means, covariance, sample_C, 3);
        [d_as_c, d_as_d, d_as_e] = getGEDConfusionMtx(means, covariance, sample_D, 3);
        [e_as_c, e_as_d, e_as_e] = getGEDConfusionMtx(means, covariance, sample_E, 3);
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
        
    elseif (method == 'MAP')
        [c_as_c, c_as_d, c_as_e] = getMAPConfusionMtx2(sample_C, sample_D, sample_E, means{1}, means{2}, means{3}, sample_C);
        [d_as_c, d_as_d, d_as_e] = getMAPConfusionMtx2(sample_C, sample_D, sample_E, means{1}, means{2}, means{3}, sample_D);
        [e_as_c, e_as_d, e_as_e] = getMAPConfusionMtx2(sample_C, sample_D, sample_E, means{1}, means{2}, means{3}, sample_E);
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
        
    elseif (method == '1NN')
        [c_as_c, c_as_d, c_as_e] = getNNConfusionMtx(1, 1, {sample_C, sample_D, sample_E}, 3);
        [d_as_c, d_as_d, d_as_e] = getNNConfusionMtx(1, 2, {sample_C, sample_D, sample_E}, 3);
        [e_as_c, e_as_d, e_as_e] = getNNConfusionMtx(1, 3, {sample_C, sample_D, sample_E}, 3);
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];

    elseif (method == 'KNN')
        [c_as_c, c_as_d, c_as_e] = getNNConfusionMtx(5, 1, {sample_C, sample_D, sample_E}, 3);
        [d_as_c, d_as_d, d_as_e] = getNNConfusionMtx(5, 2, {sample_C, sample_D, sample_E}, 3);
        [e_as_c, e_as_d, e_as_e] = getNNConfusionMtx(5, 3, {sample_C, sample_D, sample_E}, 3);
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
end
end

function [count1, count2, count3] = getGEDConfusionMtx(means, cov, sample, caseNum)
    invA = inv(cov{1});
    invB = inv(cov{2});
    
    meanA = means{1};
    meanB = means{2};
    
    count1 = 0;
    count2 = 0;
    count3 = 0;
    if (caseNum == 3)
        invC = inv(cov{3});
        meanC = means{3};
    end
    
    for i = 1:length(sample)
        pnt = sample(i, :);
        d1 = (pnt - meanA) * invA * (pnt - meanA)';
        d2 = (pnt - meanB) * invB * (pnt - meanB)';
        if (caseNum == 3)
            d3 = (pnt - meanC) * invC * (pnt - meanC)';
            if (d1 < d2 && d1 < d3)
                count1 = count1 + 1;
            elseif (d2 < d1 && d2 < d3)
                count2 = count2 + 1;
            elseif (d3 < d1 && d3 < d2)
                count3 = count3 + 1;
            end
        elseif (caseNum == 2)
            if (d1 <= d2)
                count1 = count1 + 1;
            elseif (d1 > d2)
                count2 = count2 + 1;
            end
        end
    end
    
end

function [count1, count2, count3] = getNNConfusionMtx(k, classNum, samples, caseNum)
    testSample = samples{classNum};
    count1 = 0;
    count2 = 0;
    count3 = 0;
    
    for i = 1:length(testSample)
        setDistances = zeros(1, caseNum);
        samplePnt = testSample(i, :);
        
        for j = 1:caseNum
            setDistances(:, j) = getSetDistance(k, samplePnt, samples{j}); 
        end

        min_distance = min(setDistances);

        for j = 1:caseNum
            if (min_distance == setDistances(:, j))
                if (j == 1)
                    count1 = count1 + 1;
                elseif (j == 2)
                    count2 = count2 + 1;
                elseif (j == 3)
                    count3 = count3 + 1;
                end
            end
        end
    end
end

% Calculate Euclidean distance
function distance = getEuclideanDistance(p1, p2)
    distance = sqrt((p2(:, 1) - p1(:, 1))^2 + (p2(:, 2) - p1(:, 2))^2);
end

function setDistance = getSetDistance(k, point, Set)
    
    if (k == 1)
        % Nearest Neighbor
        setDistance = 10e7; % random big number to compare to
        for i = 1:size(Set, 1)
            currentPoint = Set(i, :);
            curent_distance = getEuclideanDistance(currentPoint, point);
            if (curent_distance < setDistance)
                setDistance = curent_distance;
            end
        end
    elseif (k > 1)
        % K-Nearest Neighbor
        distances = zeros(size(Set(:, 1)));
        for i = 1:size(Set, 1)
            currentPoint = Set(i, :);
            distances(i, :) = getEuclideanDistance(currentPoint, point);
        end
        distances = sort(distances);
        setDistance = sum(distances(1:k, 1)) / k;
    end
end

function confusionMtx = getMAPConfusionMtx(means, covariance, true_stat)   
    sampleA = true_stat{1};
    sampleB = true_stat{2};
    Pa = length(sampleA)/(length(sampleA) + length(sampleB));
    Pb = length(sampleB)/(length(sampleA) + length(sampleB));
    
    map_AB = generate_MAP_db(means{1}, means{2}, covariance{1}, covariance{2}, Pa, Pb, sampleA(:,1), sampleA(:,2));
    map_BA = generate_MAP_db(means{2}, means{1}, covariance{2}, covariance{1}, Pa, Pb, sampleB(:,2), sampleB(:,2));

    s = sign(map_AB);
    a_as_a = sum(s(:)==-1);
    a_as_b = sum(s(:)==1);

    s = sign(map_BA);
    b_as_b = sum(s(:)==-1);
    b_as_a = sum(s(:)==1);
    confusionMtx = [a_as_a, a_as_b; b_as_a, b_as_b];
end

function [counter_c, counter_d, counter_e, unclassified] = getMAPConfusionMtx2(Sc, Sd, Se, Mc, Md, Me, X)   
    % X is the class we are trying to classify
    counter_c = 0;
    counter_d = 0;
    counter_e = 0;
    unclassified = 0;  % if there are any exceptions
    
    Pc = length(Sc)/(length(Sc) + length(Sd) + length(Se));
    Pd = length(Sd)/(length(Sc) + length(Sd) + length(Se));
    Pe = length(Se)/(length(Sc) + length(Sd) + length(Se));
    
    %check distance between c and d
    rows = size(X, 1);
    for i=1:rows
        xBar = X(i,:);
        Q0 = inv(Sc) - inv(Sd);
        Q1 = 2*(Md*inv(Sd)-Mc*inv(Sc));
        Q2 = Mc*inv(Sc)*Mc' - Md*inv(Sd)*Md';
        Q3 = log(Pd/Pc);
        Q4 = log(det(Sc)/det(Sd));
        distance_1 = xBar*Q0*xBar' + Q1*xBar'+Q2+2*Q3+Q4; 

        %Classified as class c
        if distance_1 <= 0
            %now compare with class C and E
            Q0 = inv(Sc) - inv(Se);
            Q1 = 2*(Me*inv(Se) - Mc*inv(Sc));
            Q2 = Mc*inv(Sc)*Mc' - Me*inv(Se)*Me';
            Q3 = log(Pe/Pc);
            Q4 = log(det(Sc)/det(Se));
            distance_2 = xBar*Q0*xBar' + Q1*xBar'+Q2+2*Q3+Q4;

            if distance_2 <= 0 
                %classifies as C
                counter_c = counter_c + 1;
            elseif distance_2 > 0
                counter_e = counter_e + 1;                
            else 
                unclassified = unclassified+1;
            end 

        else
            %Is not classifed as class c
            Q0 = inv(Sd) - inv(Se);
            Q1 = 2*(Me*inv(Se) -Md*inv(Sd));
            Q2 = Md*inv(Sd)*Md' - Me*inv(Se)*Me';
            Q3 = log(Pe/Pd);
            Q4 = log(det(Sd)/det(Se));
            
            %check if the point is classified as class D or E
            distance_3 = xBar*Q0*xBar' + Q1*xBar'+Q2+2*Q3+Q4;

            if distance_3 <= 0 
                % classified as class D
                counter_d = counter_d + 1;
            elseif distance_3 > 0
                counter_e = counter_e + 1;
            else
                 unclassified = unclassified+1;
            end 
        end
    end        
end
end