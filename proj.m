%%This set was used to make Monte Carlo estimate.
clear all
%set initial conditions
x01=10;
x02=20;
x03=30;
x04=40;
x05=50;
x06=80;
beta=0.8;
k=50;%K can also be 20, 200, 2000.
T=10^3*k;
%V is used to estimate the value function
V=[];
%M is used to generate the set we want.
M=[];
%for k=1:20
%     x1=x01;
%     x2=x02;
%     x3=x03;
%     x4=x04;
%     x5=x05;
%     x6=x06;
%     x=[x1;x2;x3;x4;x5;x6];
%     v=0;
%     for j=0:T
%         d=random('Poisson',17);
%         profit=min(90,d)-0.4*max(90-x6-x5-x4-x3-x2-x1,0)-0.1*max(x1-d,0)-0.1*(x2+x3+x4+x5+x6);
%         v=v+(beta^j)*profit;
%         x1=max(x2-max(d-x1,0),0);
%         x2=max(x3-max(d-x1-x2,0),0);
%         x3=max(x4-max(d-x1-x2-x3,0),0);
%         x4=max(x5-max(d-x1-x2-x3-x4,0),0);
%         x5=max(x6-max(d-x1-x2-x3-x4-x5,0),0);
%         x6=max(90-max(d,x1+x2+x3+x4+x5+x6),0);
%     end
%     M=[M,x];
%     V=[V,v];
% end

%%Simulation based on Monte Carlo Method. 
vmean=mean(V);
episode=1000;
count=0
epsilon=1.96*std(V)/sqrt(episode);
for count=0:(episode-1)
    count=count+1;
    x1=x01;
    x2=x02;
    x3=x03;
    x4=x04;
    x5=x05;
    x6=x06;
    v=0;
    m=0
    for j=1:k
        d=random('Poisson',17);
        %generate profit function
        profit=min(90,d)-0.4*(90-x6-x5-x4-x3-x2-x1)-0.1*max(x1-d,0)-0.1*(x1+x2+x3+x4+x5+x6);
        v=v+beta^j*profit;
        x1=max(x2-max(d-x1,0),0);
        x2=max(x3-max(d-x1-x2,0),0);
        x3=max(x4-max(d-x1-x2-x3,0),0);
        x4=max(x5-max(d-x1-x2-x3-x4,0),0);
        x5=max(x6-max(d-x1-x2-x3-x4-x5,0),0);
        x6=max(90-max(d,x1+x2+x3+x4+x5+x6),0);
    end
    V=[V,v];
    vmean=mean(V);
    epsilon=1.96*std(V)/sqrt(episode);
end
    
    