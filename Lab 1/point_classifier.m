function [ class ] = point_classifier( cd, de, ec )
    % Calculatues which class a coordinate belongs to

    %Classes
    c = 1; d = 2; e = 3;
    if cd >= 0 && de <= 0
        class = d;
    elseif cd <= 0 && ec >= 0
        class = c;
    elseif de >= 0 && ec <= 0
        class = e;
    else
end
