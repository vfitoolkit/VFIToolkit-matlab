function sub=ind2sub_vec_homemadet(sizeA, indexvec)
% Exactly same as ind2sub_vec_homemade, except transpose of input and output

%Does much the same as ind2sub, but by default returns the vector of
%subscripts that one expects. 'vec' means that it does this for a vector of
%indexes, so the output is a matrix.

% indexvec should be a row vector

nindex=size(indexvec,2); % Number of columns in indexvec (which is a row vector)

sub=zeros(length(sizeA),nindex);
sub(1,:)=rem(indexvec-1,sizeA(1))+1;
for ii=2:length(sizeA)-1
    sub(ii,:)=rem(ceil(indexvec/prod(sizeA(1:ii-1)))-1,sizeA(ii))+1;
end
sub(length(sizeA),:)=ceil(indexvec/prod(sizeA(1:length(sizeA)-1)));

end