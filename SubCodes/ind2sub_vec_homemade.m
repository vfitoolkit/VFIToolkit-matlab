function sub=ind2sub_vec_homemade(sizeA, indexvec)

%Does much the same as ind2sub, but by default returns the vector of
%subscripts that one expects. 'vec' means that it does this for a vector of
%indexes, so the output is a matrix.

% indexvec should be a column vector

nindex=size(indexvec,1); % Number of rows in indexvec (which is a column vector)

sub=zeros(nindex,length(sizeA));
sub(:,1)=rem(indexvec-1,sizeA(1))+1;
for ii=2:length(sizeA)-1
    sub(:,ii)=rem(ceil(indexvec/prod(sizeA(1:ii-1)))-1,sizeA(ii))+1;
end
sub(:,length(sizeA))=ceil(indexvec/prod(sizeA(1:length(sizeA)-1)));

end