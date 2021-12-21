%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% entropyObjective
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute the maximum entropy objective function used in
%       discreteApproximation
%
% Usage:
%       obj = entropyObjective(lambda,Tx,TBar,q)
%
% Inputs:
% lambda    - (L x 1) vector of values of the dual problem variables
% Tx        - (L x N) matrix of moments evaluated at the grid points
%             specified in discreteApproximation
% TBar      - (L x 1) vector of moments of the underlying distribution
%             which should be matched 
% q         - (1 X N) vector of prior weights for each point in the grid.
%
% Outputs:
% obj       - scalar value of objective function evaluated at lambda
% Optional (useful for optimization routines):
% gradObj   - (L x 1) gradient vector of the objective function evaluated
%             at lambda
% hessianObj- (L x L) hessian matrix of the objective function evaluated at
%             lambda
%
% Version 1.2: June 7, 2016
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [obj,gradObj,hessianObj] = entropyObjective(lambda,Tx,TBar,q)

% Some error checking

if nargin < 4
    error('You must provide 4 arguments to entropyObjective.')
end

[L,N] = size(Tx);

if length(lambda) ~= L || length(TBar) ~= L || length(q) ~= N
    error('Dimensions of inputs are not compatible.')
end

% Compute objective function

Tdiff = Tx-repmat(TBar,1,N);
temp = q.*exp(lambda'*Tdiff);
obj = sum(temp);

% Compute gradient of objective function

if nargout > 1
    temp2 = bsxfun(@times,temp,Tdiff);
    gradObj = sum(temp2,2);
end

% Compute hessian of objective function

if nargout > 2
    hessianObj = temp2*Tdiff';
end

end