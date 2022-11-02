function s = symmetry(data)
    D = zeros(16,16);
    for i = 1:16
        l = (i-1)*16+2;
        r = i*16+1;
        D(i,:) = data(l:r);
    end
    sum = 0;
    for i = 1:16
        for j = 1:16
            sum = sum + abs(D(i,j) - D(i,17-j));
        end
    end
    s = -sum/256;