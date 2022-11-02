function plo(X,y)
    [n,~] = size(X);
    t1 = 0;
    t5 = 0;
    for i = 1:n
        if y(i) == 1
            t1 = t1 + 1;
            D1(t1,1) = (X(i,2);
            D1(t1,2) = X(i,3);
        else
            t5 = t5 + 1;
            D2(t5,1) = X(i,2);
            D2(t5,2) = X(i,3);
        end
    end
    scatter(D1(:,1),D1(:,2),'b');
    hold on;
    scatter(D2(:,1),D2(:,2),'r','x');
    xlabel('Density')
    ylabel('Symmetry')