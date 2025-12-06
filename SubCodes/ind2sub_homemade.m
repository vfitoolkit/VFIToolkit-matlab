function sub=ind2sub_homemade(sizeA, index)
% Does much the same as ind2sub, but by default returns the vector of subscripts that one expects

sub=zeros(1,length(sizeA));
sub(1)=rem(index-1,sizeA(1))+1;
for ii=2:length(sizeA)-1
    sub(ii)=rem(ceil(index/prod(sizeA(1:ii-1)))-1,sizeA(ii))+1;
end
sub(length(sizeA))=ceil(index/prod(sizeA(1:length(sizeA)-1)));

end