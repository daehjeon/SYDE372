function [mtx, p_error] = confusion_matrix(size, true_stat, means, covariance, method, prob, N)
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
        map_AB = generate_MAP_db(means{1}, means{2}, covariance{1}, covariance{2}, prob{1}, prob{2}, true_stat{1}(:,1), true_stat{1}(:,2));
        map_BA = generate_MAP_db(means{2}, means{1}, covariance{2}, covariance{1}, prob{2}, prob{1}, true_stat{2}(:,2), true_stat{2}(:,2));

        s = sign(map_AB);
        a_as_a = sum(s(:)==-1);
        a_as_b = sum(s(:)==1);
        
        s = sign(map_BA);
        b_as_b = sum(s(:)==-1);
        b_as_a = sum(s(:)==1);
        mtx = [a_as_a, a_as_b; b_as_a, b_as_b];
        p_error = (a_as_b + b_as_a)/(N{1} + N{2});
        
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
        
        [c_as_c, c_as_d, c_as_e] = getMAPConfusionMtx2(cov_C, cov_D, cov_E, mean_C, mean_D, mean_E, prob{1}, prob{2}, prob{3}, sample_C);
        [d_as_c, d_as_d, d_as_e] = getMAPConfusionMtx2(cov_C, cov_D, cov_E, mean_C, mean_D, mean_E, prob{1}, prob{2}, prob{3}, sample_D);
        [e_as_c, e_as_d, e_as_e] = getMAPConfusionMtx2(cov_C, cov_D, cov_E, mean_C, mean_D, mean_E, prob{1}, prob{2}, prob{3}, sample_E);
        
        mtx = [c_as_c, c_as_d, c_as_e; d_as_c, d_as_d, d_as_e; e_as_c, e_as_d, e_as_e];
        p_error = (c_as_d + c_as_e + d_as_c + d_as_e + e_as_c + e_as_d)/(N{1} + N{2} + N{3});
    
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


function [distance] = getMAPConfusionMtx(Sa, Sb, Ma, Mb, X, Pa, Pb)   
    rows = size(X, 1);
    distance = zeros(1, rows);
    
    %From pg.18 of slide 17 
    Q0 = inv(Sa) - inv(Sb);
    Q1 = 2*(Mb*inv(Sb) -Ma*inv(Sa));
    Q2 = Ma*inv(Sa)*Ma' - Mb*inv(Sb)*Mb';
    Q3 = log(Pb/Pa);
    Q4 = log(det(Sa)/det(Sb));
    
    for i=1:rows
        xBar = X(i,:); %each row of X
        distance(i) = xBar*Q0*xBar' + Q1*xBar'+Q2+2*Q3+Q4;       
    end   
end


function [counter_c, counter_d, counter_e, unclassified] = getMAPConfusionMtx2(Sc, Sd, Se, Mc, Md, Me, Pc, Pd, Pe, X)   
    % X is the class we are trying to classify
    counter_c = 0;
    counter_d = 0;
    counter_e = 0;
    unclassified = 0;  % if there are any exceptions 
    
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

