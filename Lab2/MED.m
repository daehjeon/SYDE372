function G = MED(a,b,sample)

    m = (a(2)-b(2)) / (a(1) - b(1));
    m_prime = -1/m;
    x = (a+b)./2;
    
    G = sample(2)-x(2)-m_prime*(sample(1) - x(1));
end