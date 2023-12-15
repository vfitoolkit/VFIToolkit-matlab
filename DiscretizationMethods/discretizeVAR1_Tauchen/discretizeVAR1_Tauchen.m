function [z_grid, P]=discretizeVAR1_Tauchen(Mew,Rho,SigmaSq,znum,Tauchen_q, tauchenoptions)
% Create states vector and transition matrix for the discrete markov process approximation of 
% M-variable VAR(1) process:
%      z'=mew+rho*z+e, e~N(0,SigmaSq), by Tauchens method
%
% Inputs
%   Mew          - (M x 1) constant vector
%   Rho          - (M x M) matrix of impact coefficients
%   SigmaSq      - (M x M) variance-covariance matrix of the innovations
%                - OR M-by-1 column vector [Is assumed that covariances are zero].
%   znum         - M-by-1 column vector, or scalar; Desired number of discrete points in each dimension
%   Tauchen_q    - M-by-1 column vector, or scalar; (Hyperparameter) Defines max/min grid points as zmean+-sigma*sigmaz (I suggest 2 or 3)
%   znum         - M-by-1 vector, number of states in discretization for each variable in z
% Optional inputs (tauchenoptions)
%   parallel: set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - sum(znum)-by-1 column vector; stacked column vector containing the sum(znum) states of 
%                    the discrete approximation of z for each of the n variables
%   P              - prod(znum)-by-prod(znum) matrix; transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
%%%%%%%%%%%%%%%
% Original paper:
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"
%
% Note: This is not really the Tauchen method.
% Tauchen suggests using product of the joint marginal cdfs, but here
% instead just use the multivariate cdf directly. (If Sigma is diagonal the
% two coincide, but otherwise this code is really an modified version of
% Tauchen). So this 'is' Tauchen method, in the sense of allocating
% probabilities to grid points based on evaluating the cdf at the midpoints
% between grid points, but the way this is actually implemented is not how
% Tauchen does it.


if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
    tauchenoptions.verbose=1;
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if isfield(tauchenoptions,'parallel')==0
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(tauchenoptions,'verbose')==0
        tauchenoptions.verbose=1;
    end
end

% Turn znum and Tauchen_q into column vectors (if they aren't already)
if size(znum,2)>1 && size(znum,1)==1
    znum=znum';
end
if isscalar(znum)
    znum=znum*ones(length(znum),1);
end
if size(Tauchen_q,2)>1 && size(Tauchen_q,1)==1
    Tauchen_q=Tauchen_q';
end
if isscalar(Tauchen_q)
    Tauchen_q=Tauchen_q*ones(length(znum),1);
end

l_z=length(znum); % number of variables in VAR

%% Make sure this is a stationary VAR
if max(eig(Rho))>=1
    error('Rho means this VAR is non-stationary (so obviously cannot discretize it)')
end


%% Sigma can be a matrix or column vector.
% These two cases are treated seperately.

if size(SigmaSq,1)==size(SigmaSq,2) % Matrix
    % good
else % it is a column vector
    SigmaSq=diag(SigmaSq);
end

zmean=((eye(l_z,l_z)-Rho)^(-1))*Mew; % Mean value of z itself

% Need the std dev of z (rather than the innovations to z)
% For an AR(1) this would just be: SigmaSq = 1/(1-Rho^2);
C1 = chol(SigmaSq,'lower');
A1 = C1\(Rho*C1);
SigmaSqz = reshape(((eye(l_z^2)-kron(A1,A1))\eye(l_z^2))*reshape(eye(l_z),l_z^2,1),l_z,l_z); % unconditional variance


sigmaz=sqrt(diag(SigmaSqz)); % std dev of z
q_sigmaz = Tauchen_q.*sigmaz;


%% Construct grids
P=zeros(prod(znum),prod(znum)); % preallocate

% omega=q_sigmaz./((znum-1)/2); % The spacing between grid points
% (below codes allows for non-even spacing, even though that is redundant here)

z_grid=zeros(sum(znum),1); % Stacked column vector
% Create z_grid for each of the M dimensions
cumsum_znum=cumsum(znum);
z_grid(1:cumsum_znum(1))=linspace(-1,1,znum(1))*q_sigmaz(1)+zmean(1); % ii=1
for z_c=2:l_z
    z_grid(cumsum_znum(z_c-1)+1:cumsum_znum(z_c))=linspace(-1,1,znum(z_c))*q_sigmaz(z_c)+zmean(z_c); % For the ii-th dimension, points evenly spaced from -q_sigmaz(ii) to +q_sigmaz(ii)
end
z_gridvals=CreateGridvals(znum,z_grid,1);

if l_z>=1
    z1_grid=z_grid(1:znum(1));
    z1_gridspacing_up=[(z1_grid(2:end)-z1_grid(1:end-1))/2; Inf];
    z1_gridspacing_down=[Inf; (z1_grid(2:end)-z1_grid(1:end-1))/2]; % Note: will be subtracted from grid point, hence Inf, not -Inf
    if l_z>=2
        z2_grid=z_grid(znum(1)+1:sum(znum(1:2)));
        z2_gridspacing_up=[(z2_grid(2:end)-z2_grid(1:end-1))/2; Inf];
        z2_gridspacing_down=[Inf; (z2_grid(2:end)-z2_grid(1:end-1))/2];
        if l_z>=3
            z3_grid=z_grid(sum(znum(1:2))+1:sum(znum(1:3)));
            z3_gridspacing_up=[(z3_grid(2:end)-z3_grid(1:end-1))/2; Inf];
            z3_gridspacing_down=[Inf; (z3_grid(2:end)-z3_grid(1:end-1))/2];
            if l_z>=4
                z4_grid=z_grid(sum(znum(1:3))+1:sum(znum(1:4)));
                z4_gridspacing_up=[(z4_grid(2:end)-z4_grid(1:end-1))/2; Inf];
                z4_gridspacing_down=[Inf; (z4_grid(2:end)-z4_grid(1:end-1))/2];
                if l_z>=5
                    z5_grid=z_grid(sum(znum(1:4))+1:sum(znum(1:5)));
                    z5_gridspacing_up=[(z5_grid(2:end)-z5_grid(1:end-1))/2; Inf];
                    z5_gridspacing_down=[Inf; (z5_grid(2:end)-z5_grid(1:end-1))/2];
                end
            end
        end
    end
end

% Now do the actual multivariate normal cdf calculation
if l_z==1
    z_gridspacing_up=z1_gridspacing_up;
    z_gridspacing_down=z1_gridspacing_down;
elseif l_z==2
    z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up],1);
    z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down],1);
elseif l_z==3
    z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up],1);
    z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down],1);
elseif l_z==4
    z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up],1);
    z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down],1);
elseif l_z==5
    z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up;z5_gridspacing_up],1);
    z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down;z5_gridspacing_down],1);
end


for z_c=1:prod(znum) % lag of z
    conditionalmean=(Mew+Rho*z_gridvals(z_c,:)')';
    P(z_c,:)=mvncdf(z_gridvals-z_gridspacing_down,z_gridvals+z_gridspacing_up,conditionalmean,SigmaSq)';
end


%%
if tauchenoptions.parallel==2 
    z_grid=gpuArray(z_grid);
    P=gpuArray(P); %(z,zprime)
end


end
