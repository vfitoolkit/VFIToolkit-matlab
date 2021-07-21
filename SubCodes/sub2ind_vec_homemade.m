function idx = sub2ind_vec_homemade(sizeA,subvec)
    %Calculates the index number for a point in matrix A (whose dimensions
    %are given by the vector sizeA, that corresponds to that with the row
    %and column subscripts in the row vector a.
    %Note: Code will cause error if A is already a vector (so a is scalar)
    % Does this for each row of subvec (which is a matrix)
    
    num_of_vars=size(subvec,2);
    
    idx=subvec(:,1);
    cumprod_sizeA=cumprod(sizeA);
    if num_of_vars>1
        for i=2:num_of_vars
            idx=idx+(subvec(:,i)-1)*cumprod_sizeA(i-1);
        end
    end

end