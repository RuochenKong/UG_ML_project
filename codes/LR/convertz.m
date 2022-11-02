function z = convertz(x1,x2,k)
    n = (k+2)*(k+1)/2;
    z = zeros(1,n);
    m = 1;
    for i = 0:k
        L1 = Leg(x1,i);
        for j = 0:k-i
            L2 = Leg(x2,j);
            z(m) = L1*L2;
            m = m + 1;
        end
    end