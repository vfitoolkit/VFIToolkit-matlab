function [C , iter_number ] = GFT_inverse_mapping ( gamma , tol_value )
% This is a lightly edited version of the code provided in the Appendix to 
% Archakov & Hansen (2021) - A New Parametrization of Correlation Matrices
% Original is page 14 of https://homepage.univie.ac.at/ilya.archakov/docs/CorNoteWebAppendix.pdf
% Explanation/illustration of how to use: https://github.com/robertdkirkby/ParametrizeCoVarianceMatrix/blob/main/CoVarMatrix.m
%
% Inputs:
%     gamma: vector that parametrizes the correlation matrix
%     tol_value: tolerance for the iterative algorithm that computes C from gamma
% Outputs: 
%     C: n-by-n correlation matrix
%     iter_number: the number of iterations taken by the iterative algorithm
%
% If you use this, please cite 
% Archakov & Hansen (2021) - A New Parametrization of Correlation Matrices
% https://doi.org/10.3982/ECTA16910

%% Check if input is of proper format
% gamma is of suitable length and tolerance value belongs to a proper interval
n = 0.5*(1+sqrt(1+8*length(gamma)));
if ~isvector(gamma)
    error('gammm should be a vector')
end
if n ~= floor(n)
    error('gamma should have n(n-1)/2 elements, where nxn is the size of the covar matrix you are parametrizing')
end
if tol_value < 10^(-14) || tol_value  > 1^(-4)
    error('tol_value must be between 10^(-14) and 10^(-4)')
end


if n==2
    %% n=2 has analytic solution (which will be faster)
    % Fisher-transformation is F(p)=0.5*log((1+p)/(1-p))
    % We just need to do inverse of this
    p=(exp(2*gamma)-1)/(1+exp(2*gamma)); % n=1, so gamma is scalar
    C=[1,p; p, 1];
    iter_number=0; % no iterations
else
    %% Now for the actual work :)
    % Place elements from gamma into off - diagonal parts
    % and put zeros on the main diagonal of nxn symmetric matrix A
    A = zeros(n,n);
    A(logical(tril(ones(n,n),-1))) = gamma; % Put gamma into the lower triangular elements of A (the -1 is to skip the diagonal itself)
    A = A + A'; % repeat the lower diagonal as the upper diagonal

    % Preallocate
    diag_vec = diag(A); % realistically, this is just preallocating as it a vector of zeros because of what we have done with A so far
    diag_ind = logical(eye(n,n));

    %% Iterative algorithm to get the proper diagonal vector
    iter_number = -1;
    dist = sqrt(n);
    while dist > sqrt(n)*tol_value
        diag_delta = log(diag(expm(A)));
        diag_vec = diag_vec - diag_delta ;
        A (diag_ind) = diag_vec ;
        dist = norm (diag_delta);
        iter_number = iter_number + 1;
    end
    % Get a unique reciprocal correlation matrix
    C = expm(A);
    C(diag_ind) = ones(n,1);
end

end