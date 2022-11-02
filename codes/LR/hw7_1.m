function hw7_1
    bound = @(E,N,d,delta) E+ sqrt(8/N* log(4*( (2*N)^d+1 ) /delta));
    phy = @(x) [1;x(2);x(3);x(2)*x(3);
                x(2)*x(3)^2;x(3)*x(2)^2;
                x(2)^3;x(3)^3];

    Train = lf('ZipDigits.train');
    [X,y] = convert(Train);
    
    [n,~] = size(X);
    Z = zeros(n,8);
    for i = 1:n
        xn = X(i,:);
        Z(i,:) = phy(xn);
    end
    
    Xp = (transpose(X) * X) \ transpose(X);
    wlin = Xp * y;
    w = pocket(wlin,X,y);
    E = Error(w,X,y);
    fprintf('E_in = %f %%\n',E*100);
    [D1,D2] = seperate(X,y);
    figure(1);
    scatter(D1(:,1),D1(:,2),'b');
    hold on;
    scatter(D2(:,1),D2(:,2),'r','x');
    xlabel('Density')
    ylabel('Symmetry')
    
    a = -1:0.001:0.4;
    b = -( w(1)+w(2).*a)./ w(3);
    
    plot(a,b,'black');
    title('Train');
    hold off;
    
    Test = lf('ZipDigits.test');
    [Xtest,ytest] = convert(Test);
    
    Etest = Error(w,Xtest,ytest);
    fprintf('E_test = %f %%\n',Etest*100);
    figure(2);
    [D1,D2] = seperate(Xtest,ytest);
    scatter(D1(:,1),D1(:,2),'b');
    hold on;
    scatter(D2(:,1),D2(:,2),'r','x');
    xlabel('Density')
    ylabel('Symmetry')
    
    a = -1:0.001:0.4;
    b = -( w(1)+w(2).*a)./ w(3);
    
    plot(a,b,'black');
    title('Test');
    hold off;
    
    b1 = bound(E,1561,3,0.05);
    b2 = bound(Etest,424,3,0.05);
    
    fprintf('E_out bound from E_in = %f %%\n',b1*100);
    fprintf('E_out bound from E_test = %f %%\n',b2*100);
    
    