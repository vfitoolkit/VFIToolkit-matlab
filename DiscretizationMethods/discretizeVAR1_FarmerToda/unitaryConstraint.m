function [c,ceq] = unitaryConstraint(X)

c = [];
ceq = X'*X-eye(size(X,2));

end

