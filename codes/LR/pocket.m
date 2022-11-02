function what = pocket(w,X,y)
    what = w;
    E = Error(w,X,y);
    [n,~] = size(X);
    for t = 1: 10000
        for i = 1:n
            xn = transpose(X(i,:));
            if sign(transpose(what)*xn) ~= y(i)
                wnew = what + y(i)*xn;
            end
        end
        Enew = Error(wnew,X,y);
        if Enew < E
            what = wnew;    
        end
    end