function mtx = confusion_matrix(size, true_stat, means, covariance, method)
%confusion_matrix Returns the confusion matrix from the results given

if (size == 2) 
    
    sample_A = true_stat{1};
    sample_B = true_stat{2};
    
    mean_A = means{1};
    mean_B = means{2};

    if (method == 'MED')

    elseif (method == 'GED')
        [a_as_a, a_as_b, tmp] = getGEDConfusionMtx(means, covariance, sample_A, 2);
        [b_as_a, b_as_b, tmp] = getGEDConfusionMtx(means, covariance, sample_B, 2);
        mtx = [a_as_a, a_as_b; b_as_a, b_as_b];
    elseif (method == 'MAP')
    elseif (method == 'NN')
    elseif (method == 'KNN')
    end
elseif (size == 3)
    sample_C = true_stat{1};
    sample_D = true_stat{2};
    sample_E = true_stat{3};
    
    mean_C = means{1};
    mean_D = means{2};
    mean_E = means{3};
    
    cov_C = covariance{1};
    cov_D = covariance{2};
    cov_E = covariance{3};
    
    if (method == 'MED')

    elseif (method == 'GED')
        [c_as_c, c_as_d, c_as_e] = getGEDConfusionMtx({mean_C, mean_D, mean_E}, {cov_C, cov_D, cov_E}, sample_C, 3);
        [d_as_c, d_as_d, d_as_e] = getGEDConfusionMtx({mean_C, mean_D, mean_E}, {cov_C, cov_D, cov_E}, sample_D, 3);
        [e_as_c, e_as_d, e_as_e] = getGEDConfusionMtx({mean_C, mean_D, mean_E}, {cov_C, cov_D, cov_E}, sample_E, 3);
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
    elseif (method == 'MAP')
    elseif (method == 'NN')
    elseif (method == 'KNN')
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


