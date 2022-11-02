function D = lf(filename)
    fid = fopen(filename);
    line = fgetl(fid);
    t = 0;
    while ischar(line)
        tmp(:) = str2num(line);
        if tmp(1) == 1 || tmp(1) == 5
            t = t + 1;
            D(t,:) = tmp;
        end
        line = fgetl(fid);
    end
        