function hw9_6
    Train = lf('ZipDigits.train');
    [XTrain,yTrain] = convert(Train);
    [n,~] = size(XTrain);
    d = 45; % (8+2)*(8+1)/2
    
    Test = lf('ZipDigits.test');
    [Xtest,ytest] = convert(Test);
    [ntest,~] = size(Xtest);
    
    nt = n + ntest;
    X = zeros(nt,3);
    y = zeros(nt,1);
    
    X(1:n,:) = XTrain(:,:);
    X(n+1:nt,:) = Xtest(:,:);
    
    y(1:n) = yTrain(:);
    y(n+1:nt) = ytest(:);
    
    n = nt;
    z = zeros(n,d);
    for i = 1:n
        z(i,:) = convertz(X(i,2),X(i,3),8);
    end
    for i = 1:ntest
        ztest(i,:) = convertz(Xtest(i,2),Xtest(i,3),8);
    end
    

    I = eye(d,d);
    l = 0.0774;
    zp = (transpose(z)*z + l.*I)\transpose(z);
    w = zp * y;
    Etest = Error(w,ztest,ytest);
        
    
    fprintf('%8.16f\n',Etest);
    disp(n);
    
  