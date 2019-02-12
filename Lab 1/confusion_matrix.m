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
