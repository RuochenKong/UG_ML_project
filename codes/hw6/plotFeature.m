
function plotFeature(Fdata)
%plot the features

    j0=find(Fdata(:,1)~=1);
    j1=find(Fdata(:,1)==1);
    l=plot(Fdata(j0,2),Fdata(j0,3),'rx',Fdata(j1,2),Fdata(j1,3),'bo');
    h=gca;
    set(h,'FontSize',14)
    set(l(2),'MarkerSize',12);
    set(l(1),'MarkerSize',12);
    set(l,'LineWidth',2)
    leg=legend(l,'Not Digit 1','Digit 1',3);
    set(leg,'LineWidth',2,'FontSize',14);
    xh=xlabel('Intensity');set(xh,'FontSize',14);
    yh=ylabel('Symmetry');set(yh,'FontSize',14);
%axis([0 0.7 -8 0.1]);