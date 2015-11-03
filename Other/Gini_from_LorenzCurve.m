function Gini=Gini_from_LorenzCurve(LorenzCurve);
%Takes in a LorenzCurve with N evenly spaced point points from 1/N up to 1 (in practice my
%codes default to using N=100 when creating the LorenzCurve). Returns the Gini
%coefficient.

N=length(LorenzCurve);

%Use the Gini=A/(A+B)=2*A formulation for Gini coefficent (see wikipedia).
A=0;
for i=1:N
    %Note: i/N-LorenzCurve(i) is the height of 'A' in slice i, and 1/N is it's width 
    A=A+(i/N-LorenzCurve(i))*1/N;
end
Gini=2*A;

end
