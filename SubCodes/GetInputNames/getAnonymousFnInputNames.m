function names=getAnonymousFnInputNames(FnToEvaluate)

temp=func2str(FnToEvaluate); % Note that this strips out any spaces automatically

names=cell(nargin(FnToEvaluate),1);
vv=1;

v1=3; % The first two should be '@('
% Following commented lines would test for this.
% if strcmp(temp(1:2),'@(')
%     error('one of the functions appears to be incorrectly declared (cant read the variable input names) \n')
% end

for ii=3:length(temp)
    currchar=temp(ii);
    if strcmp(currchar,',')
        names{vv}=temp(v1:ii-1);
        vv=vv+1;
        v1=ii+1;
    elseif strcmp(currchar,')')
        names{vv}=temp(v1:ii-1);
        break
    end
end

end