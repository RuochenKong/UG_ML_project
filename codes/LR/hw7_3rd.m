function hw7_3rd
    phy = @(x) [1;x(2);x(3);x(2)*x(3);
                x(2)^2;x(3)^2;
                x(2)*x(3)^2;x(3)*x(2)^2;
                x(2)^3;x(3)^3];

    Train = lf('ZipDigits.train');
    [X,y] = convert(Train);
    
    [n,~] = size(X);
    Z = zeros(n,10);
    for i = 1:n
        xn = X(i,:);
        Z(i,:) = phy(xn);
    end
    disp(Z(1:5,:))
    Zp = (transpose(Z) * Z) \ transpose(Z);
    wlin = Zp * y;
    w = pocket(wlin,Z,y);
    E = Error(w,Z,y);
    fprintf('E_in = %f %%\n',E*100);
    
    
    Test = lf('ZipDigits.test');
    [Xtest,ytest] = convert(Test);
    
    [n,~] = size(Xtest);
    Ztest = zeros(n,10);
    for i = 1:n
        xn = Xtest(i,:);
        Ztest(i,:) = phy(xn);
    end
    
    
    Etest = Error(w,Ztest,ytest);
    fprintf('E_test = %f %%\n',Etest*100);
