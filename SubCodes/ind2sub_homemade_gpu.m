function sub=ind2sub_homemade_gpu(sizeA, index)

%Does much the same as ind2sub, but by default returns the vector of
%subscripts that one expects

% The following three lines are the old version of the code. Abandoned as
% it uses cells and so had problems with "codegen". Actually the new code
% is faster anyway!
% A_subscripts_cell = cell(1,length(sizeA));
% [A_subscripts_cell{:}] = ind2sub(sizeA,index);
% sub=[A_subscripts_cell{:}];

sub=zeros(1,length(sizeA),'gpuArray');
sub(1)=rem(index-1,sizeA(1))+1;
for ii=2:length(sizeA)-1
    sub(ii)=rem(ceil(index/prod(sizeA(1:ii-1)))-1,sizeA(ii))+1;
end
sub(length(sizeA))=ceil(index/prod(sizeA(1:length(sizeA)-1)));

end