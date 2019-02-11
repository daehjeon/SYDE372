function classification = point_classifier(case_size, results, x, y)
    % Calculatues which class a coordinate belongs to
    [a, b, c, d, e] = deal(1, 2, 3, 4, 5);
    classification = zeros(size(x, 1), size(y, 2));
    
    if (case_size == 2)
        ab = results{1};
    elseif (case_size == 3)
        cd = results{1};
        de = results{2};
        ec = results{3};
    end
    
    for i = 1:size(x, 1)
        for j = 1:size(y, 2)
            if (case_size == 2)
                if (ab(i, j) >= 0)
                    classification(i, j) = a;
                else
                    classification(i, j) = b;
                end
            elseif (case_size == 3)
                if (cd(i, j) >= 0 && de(i, j) <= 0)
                    classification(i, j) = d;
                elseif (de(i, j) >= 0 && ec(i, j) <= 0)
                    classification(i, j) = e;
                elseif (ec(i, j) >= 0 && cd(i, j) <= 0)
                    classification(i, j) = c;
                end      
            end
        end
    end
end
