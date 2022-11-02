function c
    X = [0 1 2 3 4 5 6 7 8 9];
    D = zeros(3,3);
    for i = 1:3
        l = (i-1)*3+2;
        r = i*3+1;
        D(i,:) = X(l:r);
    end
    
    sum = 0;
    for i = 1:3
        for j = 1:3
            disp(abs(D(i,j) - D(i,4-j)));
            sum = sum + abs(D(i,j) - D(i,4-j));
        end
    end
    disp(D);
    disp(sum);