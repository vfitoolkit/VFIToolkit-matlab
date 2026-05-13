function [Vtilde,Policy2,V]=ValueFnIter_FHorz_QuasiHyperbolicN_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% See documentation for a fuller explanation of this.

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

    Vtilde=V;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V for naive
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    Vtilde=V;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=V_Jplus1; % Note: The V_Jplus1 input should be V for naive

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);

    entireEV=kron(EV,ones(N_d,1));

    % For naive, we compute V which is the exponential
    % discounter case, and then from this we get Vtilde and
    % Policy (which is Policytilde) that correspond to the
    % naive quasihyperbolic discounter
    % First V
    entireRHS=ReturnMatrix+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    [Vtemp,~]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp;
    % Now Vtilde and Policy
    entireRHS=ReturnMatrix+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vtilde(:,N_j)=Vtemp; % Evaluate what would have done under exponential discounting
    Policy(:,N_j)=maxindex; % Use the policy from solving the problem of Vtilde
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=V(:,jj+1); % Use V (goes into the equation to determine V)

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);

    entireEV=kron(EV,ones(N_d,1));

    % For naive, we compute V which is the exponential
    % discounter case, and then from this we get Vtilde and
    % Policy (which is Policytilde) that correspond to the
    % naive quasihyperbolic discounter
    % First V
    entireRHS=ReturnMatrix+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    [Vtemp,~]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    % Now Vtilde and Policy
    entireRHS=ReturnMatrix+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vtilde(:,jj)=Vtemp; % Evaluate what would have done under exponential discounting
    Policy(:,jj)=maxindex; % Use the policy from solving the problem of Vtilde
end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
