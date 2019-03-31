function cov = get_cov(data,mu)
    s = 0;
    for i=1:size(data,1)
        s = s + (data(i,:)' - mu')*(data(i,:)' - mu')';
    end
    cov = 1/size(data,1) .* s;
end