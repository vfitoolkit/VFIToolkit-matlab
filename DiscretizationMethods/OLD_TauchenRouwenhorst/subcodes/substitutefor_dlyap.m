function P = substitutefor_dlyap(A,X)
% Replicates functionality of dlyap function from Matlabs Control System
% Toolbox. Slower than dlyap, but seems to work fine. 
% Original code copied from https://in.mathworks.com/matlabcentral/newsreader/view_thread/16018
%
% Solves the lyapunov equation P - A'*P*A = X by transforming A & X matrices
% to complex Schur form, computes the solution of the resulting triangular
% system, and transforms this solution back. A and X are square matrices.
%%%%%%%

% Transform the matrix A to complex Schur form
% A = U * T * U', U*U' = I
[U,T] = schur(complex(A)); %real schur form since A is real
%Now: P - (U*T'*U')*P*(U*T*U') = X which means
%U'*P*U - (T'*U')*P*(U*T) = U'*X*U
%Let Q = U'*P*U yields, Q - T'*Q*T = U'*X*U = Y

% Solve for Q = U'*P*U by transforming X to Y = U'*X*U
% Therefore, solve: Q - T*Q*T' = Y. Save memory by using P for Q.
dim = size(A(:,1));
Y = U' * X * U;
P = Y; %Initialize P
T1 = T;
T2 = T';
for col = dim:-1:1,
    for row = dim:-1:1,
        P(row,col) = P(row,col) + T1(row,row+1:dim)*(P(row+1:dim,col+1:dim)*T2(col+1:dim,col));
        P(row,col) = P(row,col) + T1(row,row)*(P(row,col+1:dim)*T2(col+1:dim,col));
        P(row,col) = P(row,col) + T2(col,col)*(T1(row,row+1:dim)*P(row+1:dim,col));
        P(row,col) = P(row,col) / (1 - T1(row,row)*T2(col,col));
    end
end
%U*P*U' - U*T1*P*T1'*U' - X
% Convert Q to P by P = U'*Q*U.
P = U*P*U';
%P - A*P*A' - X