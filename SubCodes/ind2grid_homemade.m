function gridvals=ind2grid_homemade(sizeA, index, grid)

%first block of code is just a repeat of contents of sub=ind2sub_homemade(sizeA,ind);
sub=zeros(1,length(sizeA));
sub(1)=rem(index-1,sizeA(1))+1;
for ii=2:length(sizeA)-1
    sub(ii)=rem(ceil(index/prod(sizeA(1:ii-1)))-1,sizeA(ii))+1;
end
sub(length(sizeA))=ceil(index/prod(sizeA(1:length(sizeA)-1)));


if length(sizeA)>1
    sub=sub+[0,cumsum(sizeA(1:end-1))];
end
gridvals=grid(sub);
    
end