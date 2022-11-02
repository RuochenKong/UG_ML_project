function e = Error(w,X,y)
    [n,~] = size(X);
    %tranX = transpose(X);
    %tranw = transpose(w);
    %trany = transpose(y);
    %r = tranw*tranX*X*w - 2.*tranw*tranX*y + trany*y;
    %e = r/n;
    
    Y = X*w;
    sum = 0;
    for i = 1: n
        if sign(Y(i)*y(i)) < 0
            sum = sum + 1;
        end
    end
    e = sum/n;