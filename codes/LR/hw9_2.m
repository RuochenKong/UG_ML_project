function hw9_2
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
    
    
    lambda = 0.0774;
    I = eye(d,d);
    zp = (transpose(Ztrain)*Ztrain + lambda.*I)\transpose(Ztrain);
    w = zp * yt;
    
    
    
    x =-1:0.005:0.4;
    n=length(x);
    r = 1; b = 1;
    xr1 = zeros(1,100); xr2 = zeros(1,100);
    xb1 = zeros(1,100); xb2 = zeros(1,100);
    for i = 1:n
        for j = 1:n
            tmpz = convertz(x(i),x(j),8);
            tmpy = transpose(w)*transpose(tmpz);
            if tmpy >= 0
                xr1(r) = x(i); xr2(r) = x(j); r = r + 1;
            else
                xb1(b) = x(i); xb2(b) = x(j); b = b + 1;
            end
        end
    end
   
    scatter(xr1,xr2,'MarkerFaceAlpha',0.1);
    hold on;
    scatter(xb1,xb2,'MarkerFaceAlpha',0.1);
    
    [D1,D2] = seperate(XTrain,yTrain);
    scatter(D1(:,1),D1(:,2),'b');
    scatter(D2(:,1),D2(:,2),'white','x');
    xlabel('Density')
    ylabel('Symmetry')
    legend('g(x) = +1','g(x) = -1','Digit 1','Digit 5')
    
    [D1,D2] = seperate(Xtest,ytest);
    scatter(D1(:,1),D1(:,2),'b');
    scatter(D2(:,1),D2(:,2),'white','x');
    xlabel('Density')
    ylabel('Symmetry')
    legend('g(x) = +1','g(x) = -1','Digit 1','Digit 5')
    hold off;
  