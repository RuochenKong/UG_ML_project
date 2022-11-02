function [X,y] = convert(Data)
    [n,~] = size(Data);
    X = ones(n,3);
    y = zeros(n,1);
    
    for i = 1: n
        X(i,2) = density(Data(i,:));
        X(i,3) = symmetry(Data(i,:));
        if Data(i,1) == 1
            y(i) = 1;
        else
            y(i) = -1;
        end
    end
    