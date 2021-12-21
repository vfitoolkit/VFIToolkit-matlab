function [U,fval] = minVarTrace(A)
% find a unitary matrix U such that the diagonal components of U'*AU is as
% close to a multiple of identity matrix as possible
warning off

[s1,s2] = size(A);
if s1 ~= s2
    error('input matrix must be square')
end

K = s1; % size of A
d = trace(A)/K; % diagonal of U'*A*U should be closest to d
obj =@(X)(norm(diag(X'*A*X)-d));
options = optimoptions(@fmincon,'Display','off');
[U,fval] = fmincon(obj,eye(K),[],[],[],[],[],[],@unitaryConstraint,options);

warning on

end

