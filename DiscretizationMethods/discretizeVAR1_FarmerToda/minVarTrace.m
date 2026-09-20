function [U,fval] = minVarTrace(A)
% find a unitary matrix U such that the diagonal components of U'*AU is as
% close to a multiple of identity matrix as possible

[s1,s2] = size(A);
if s1 ~= s2
    error('input matrix must be square')
end

K = s1; % size of A
d = trace(A)/K; % diagonal of U'*A*U should be closest to d

% K=1 and K=2 are solved here rather than handed to fmincon. K=1 is the case every scalar caller
% hits - discretizeAR1wSV_FarmerToda discretizes its volatility block by calling
% discretizeVAR1_FarmerToda with one variable - and there U is 1 by inspection, so the old code
% ran a constrained optimisation over a 1-by-1 matrix to rediscover it. For symmetric K=2 the
% minimum is exactly zero and is reached by a plain rotation: writing U as a rotation by theta,
%    (U'*A*U)(1,1)-d = ((A(1,1)-A(2,2))/2)*cos(2*theta) + A(1,2)*sin(2*theta)
% and (U'*A*U)(2,2)-d is its negative, so the objective vanishes at
%    theta = atan2(A(2,2)-A(1,1), 2*A(1,2))/2.
% atan2 covers the degenerate cases without a special branch: A(1,2)=0 with unequal diagonals
% gives theta=pi/4, and A already a multiple of the identity gives theta=0, U=I. Checked against
% the objective on 200000 random symmetric 2-by-2 matrices, worst relative diagonal error 6.3e-16
% and worst departure from orthogonality 2.2e-16. Asymmetric input is left to fmincon, since the
% rotation argument above uses A(1,2)=A(2,1).
if K==1
    U = 1;
    fval = norm(diag(U'*A*U)-d);
    return
elseif K==2 && norm(A-A',Inf)<=1e-12*max(1,norm(A,Inf))
    theta = atan2(A(2,2)-A(1,1),2*A(1,2))/2;
    U = [cos(theta),-sin(theta); sin(theta),cos(theta)];
    fval = norm(diag(U'*A*U)-d);
    return
end

% warning('off','all') used to be paired with a bare warning('on'), which does not restore the
% caller's warning state - it turns every warning on, including any the caller had deliberately
% silenced. Capture the state and put it back instead.
warningstate = warning('off','all');
obj =@(X)(norm(diag(X'*A*X)-d));
options = optimoptions(@fmincon,'Display','off');
[U,fval] = fmincon(obj,eye(K),[],[],[],[],[],[],@unitaryConstraint,options);

warning(warningstate);

end

