function plotEllipsis(cov, mean, cluster)    
    
[eigenVec,eigenVal] = eig(cov);
[index, r] = find(eigenVal == max(max(eigenVal)));

% Get largest eigenVec
maxEigenVec = eigenVec(:, index);
maxEigenVal = max(max(eigenVal));

%Find smallest eigenVec 
if (index==1)
    minEigenVal=max(eigenVal(:,2));
    minEigenVec=eigenVec(:,2);
else
    minEigenVal=max(eigenVal(:,1));
    minEigenVec=eigenVec(1,:);
end

% Get angle 
theta = atan2(maxEigenVec(2), maxEigenVec(1));

%
% Plot_Ellipse(x,y,theta,a,b)
%
% This routine plots an ellipse with centre (x,y), axis lengths a,b
% with major axis at an angle of theta radians from the horizontal.
%
% *** Note: many students had a LOT of trouble with this.
%           I suggest you take a little time to look at this routine
%           to see how it works.  It's only four lines of code, but
%           some people spent hours trying to write something like
%           this themselves last year.
%

%
% Author: P. Fieguth
%         Jan. 98
%

np = 100;
ang = [0:np]*2*pi/np;
pts = [mean(1);mean(2)]*ones(size(ang)) + [cos(theta) -sin(theta); sin(theta) cos(theta)]*[cos(ang)*sqrt(maxEigenVal); sin(ang)*sqrt(minEigenVal)];
plot( pts(1,:), pts(2,:), 'LineWidth', 3 );
hold on
scatter(cluster(:, 1), cluster(:, 2));

end