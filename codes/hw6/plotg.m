function p = plotg(X) 
    D = zeros(16,16);
    for i = 1:16
        l = (i-1)*16+2;
        r = i*16+1;
        D(i,:) = X(7,l:r);
    end
    
    p = displayimage(D);