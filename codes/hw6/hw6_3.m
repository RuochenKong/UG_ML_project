function hw6_3
    X = lf('ZipDigits.test');
    [n,~] = size(X);
    t1 = 0;
    t5 = 0;
    for i = 1:n
        if X(i) == 1
            t1 = t1 + 1;
            D1(t1,1) = symmetry(X(i,:));
            D1(t1,2) = density(X(i,:));
        else
            t5 = t5 + 1;
            D2(t5,1) = symmetry(X(i,:));
            D2(t5,2) = density(X(i,:));
        end
    end
    X = lf('ZipDigits.train');
    [n,~] = size(X);
    for i = 1:n
        if X(i) == 1
            t1 = t1 + 1;
            D1(t1,1) = symmetry(X(i,:));
            D1(t1,2) = density(X(i,:));
        else
            t5 = t5 + 1;
            D2(t5,1) = symmetry(X(i,:));
            D2(t5,2) = density(X(i,:));
        end
    end
    scatter(D1(:,2),D1(:,1),'b');
    hold on;
    scatter(D2(:,2),D2(:,1),'r','x');
    xlabel('Density')
    ylabel('Symmetry')
    title('train + test')
    hold off;