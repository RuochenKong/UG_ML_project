function hw9_4
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
    
    Ztrain = z(201:500,:);
    yt = y(201:500);
    ztest = zeros(1685,45);
    ytest = zeros(1685,1);
    ztest(1:200,:) = z(1:200,:);
    ztest(201:1685,:) = z(501:1985,:);
    ytest(1:200) = y(1:200);
    ytest(201:1685) = y(501:1985);
    lamda = 0:0.01:2;
    I = eye(d,d);
    [~,numl] = size(lamda);
    Etest = zeros(1,numl);
    Ecv = zeros(1,numl);
    for i = 1:numl
        l = lamda(i);
        zp = (transpose(Ztrain)*Ztrain + l.*I)\transpose(Ztrain);
        w = zp * yt;
        Etest(i) = Error(w,ztest,ytest);
        
        H = Ztrain*zp;
        yhat = H*yt;
        Ecv(i) = 0;
        for j = 1:300
            Ecv(i) = Ecv(i) + ((yhat(j)-yt(j))/(1-H(j,j)))^2;
        end
        Ecv(i) = Ecv(i)/300;
    end
    
    scatter(lamda,Etest,'.');
    hold on;
    scatter(lamda,Ecv,'.');
    hold off;
    
    
    xlabel('lambda')
    ylabel('Error')
    
    legend('Etest','Ecv')
    
    
  