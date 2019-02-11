function [ dist ] = generateMED_db( mean_a, mean_b, X, Y )
    dist = zeros(size(X, 1), size(Y, 2));
    get_dist = @(point, mean) (point-mean) * (point-mean)';

    for i=1:size(X, 1)
        for j=1:size(Y, 2)
            point = [X(i, j) Y(i, j)];
            dist(i, j) = get_dist(point, mean_a) - get_dist(point, mean_b);
        end
    end
end
