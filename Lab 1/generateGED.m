function [distance] = generateGED(sigma1, mu1, sigma2, mu2, x, y, distanceAB)
    %For class A & B    
    if distanceAB > 0
        % Transposing the covariances 
        invA = inv(sigma1);
        invN = inv(sigma2);
        distance = distanceAB;
        get_dist = @(point, covar, mean) sqrt((point - mean) * inv(covar) * transpose(point - mean));

        % MICD/GED equation
        for (i=1:size(x,1))
            for(j=1:size(y,2))
                xBar = [x(i,j) y(i,j)];
                distance(i, j) = get_dist(xBar, sigma1, mu1) - get_dist(xBar, sigma2, mu2);
            end
        end  
        
    %For class C, D, E    
    else
        dist = zeros(size(x, 1), size(y, 2));
        get_dist = @(point, covar, mean) sqrt((point - mean) * inv(covar) * transpose(point - mean));

        for i=1:size(x, 1)
            for j=1:size(y, 2)
                xBar = [x(i,j) y(i,j)];
                distance(i, j) = get_dist(xBar, sigma1, mu1) - get_dist(xBar, sigma2, mu2);
            end
        end
    end
end