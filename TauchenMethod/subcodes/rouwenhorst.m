function P=rouwenhorst(h,p,q)

if h==2
    P=[p 1-p; 1-q q];
else
    P1=rouwenhorst(h-1,p,q);
    z=zeros(1,h);
    z1=zeros(h-1,1);
    P=[p*P1 z1; z]+[z1 (1-p)*P1; z]+...
        [z; (1-q)*P1 z1]+[z; z1 q*P1];
    P(2:h-1,:)=P(2:h-1,:)/2;
end

end