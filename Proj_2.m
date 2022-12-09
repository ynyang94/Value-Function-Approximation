%%This document was used to generate the data set.
clear all
%set initial conditions
x01=10; 
x02=20; 
x03=30;
x04=40;
x05=50;
x06=80;
%discount rate
beta=0.8;
%the size we need generate
k=50;
T=50000;
%V is used for calculating the value function
V=[];
%M is used for storing the data we generate.
M=[];
x1=x01;
x2=x02;
x3=x03;
x4=x04;
x5=x05;
x6=x06;
v=0;
for j=1:50000
    %decide the initial size of Markov Chain. Here is the size for Q2 (b)
    d=random('Poisson',17);
    %generate the profit function
    profit=min(90,d)-0.4*max(90-x6-x5-x4-x3-x2-x1,0)-0.1*max(x1-d,0)-0.1*(x2+x3+x4+x5+x6);
    v=v+(beta^j)*profit;
    x1=max(x2-max(d-x1,0),0);
    x2=max(x3-max(d-x1-x2,0),0);
    x3=max(x4-max(d-x1-x2-x3,0),0);
    x4=max(x5-max(d-x1-x2-x3-x4,0),0);
    x5=max(x6-max(d-x1-x2-x3-x4-x5,0),0);
    x6=max(90-max(d,x1+x2+x3+x4+x5+x6),0);
    m=[x1;x2;x3;x4;x5;x6];
    %collect the data set we need.
    if mod(j,1000)==0
        M=[M,m]
    end
end
%After generating, use the command "xlswrite(filename, variable) to collect
%the data set.
