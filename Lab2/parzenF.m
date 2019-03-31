function Px = parzenF(X,x,h)
    
    N = length(X);
%     x_repeated = repmat(x,1,N);
    x=x';
    for i=1:N
        s(i) = (1/N * 1/h *  1/(sqrt(2*pi))*exp(-(x-X(i,:))*(x-X(i,:))'./(2*h.^2)));
    end
    Px=sum(s);
end