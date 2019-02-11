function classification = generateNN_db(k, Sets)
%generateNN_db Generate the decision boundary using Nearest Neighbor\

% Define point range and resolution
numOfSets = length(Sets);
[x_lower, x_higher, y_lower, y_higher] = deal(0,0,0,0);
for i = 1:numOfSets
    [currentSet] = Sets{i};
    if (min(currentSet(:, 1)) < x_lower)
        x_lower = min(currentSet(:, 1));
    end
    if (max(currentSet(:, 1)) > x_higher)
        x_higher = max(currentSet(:, 1));
    end
    if (min(currentSet(1, :)) < y_lower)
        y_lower = min(currentSet(1, :));
    end
    if (max(currentSet(1, :)) > y_higher)
        y_higher = max(currentSet(1, :));
    end
end

resolution = 0.05; % Resolution for grid
[gridX, gridY] = meshgrid(x_lower:resolution:x_higher, y_lower:resolution:y_higher);

classification = zeros(size(gridX, 1), size(gridY, 2));

for x = 1:size(gridX, 1)
    for y = 1:size(gridX, 2)
        setDistances = zeros(1, numOfSets);
        samplePnt = [gridX(x, y) gridY(x, y)];
        for i = 1:numOfSets
            setDistances(:, i) = getSetDistance(k, samplePnt, Sets{i}); 
        end
       
        min_distance = min(setDistances);
        
        for i = 1:numOfSets
            if (min_distance == setDistances(:, i))
                classification(x, y) = i;
            end
        end
    end
end

contour(gridX, gridY, classification);

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

