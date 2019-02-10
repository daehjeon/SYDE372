function [db] = generateNN_db(S_1, S_2)
%generateNN_db Generate the decision boundary using Nearest Neighbor

% Define point range and resolution
% x_lower = min(min(S_1(1, :)), min(S_2(1, :)));
% x_higher = max(max(S_1(1, :)), max(S_2(1, :)));
% y_lower = min(min(S_1(:, 1)), min(S_2(:, 1)));
% y_higher = max(max(S_1(:, 1)), max(S_2(:, 1)));

% Temp settings for bounds, change to the lines above late
x_lower = -5;
x_higher = 20;
y_lower = 0;
y_higher = 20;

% TODO: maybe adjust these to improve the db
resolution = 0.2;
sensitivity = 0.01;
tolerance = 0.6;

db = [];
% This is gonna be a bitch to run, might need to optimize
for x = x_lower:+resolution:x_higher
    for y = y_lower:+resolution:y_higher
        % Calculate the minimum distance between sample and S_1 points
        d_1_min = 10e5; % random big number to compare to
        for i = 1:size(S_1, 1)
            point1 = S_1(i, :);
            d_1_curr = (point1(:, 1) - x)^2 + (point1(:, 2) - y)^2;
            if (d_1_curr < d_1_min)
                d_1_min = d_1_curr;
                temp_point = point1;
            end
        end
        
        % Calculate distance between sample and points in Set 2 (S_2), 
        % Check if any points give the same min distance
        for j = 1:size(S_2, 1)
            point2 = S_2(j, :);
            d_2_curr = (point2(:, 1) - x)^2 + (point2(:, 2) - y)^2;
            % Store point in db if the difference is small enough
            if (abs(d_2_curr - d_1_min) < sensitivity && d_1_min < tolerance)
                db = [db; [x, y]];
            end
        end
    end
end

end

