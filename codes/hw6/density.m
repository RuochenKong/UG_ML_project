function d = density(data)
    sum = 0;
    for i = 2:257
        sum = sum + data(i);
    end
    d = sum/256;