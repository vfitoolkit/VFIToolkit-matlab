function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_noz_e_raw(n_a,n_e,N_j, a_grid,e_gridvals_J,pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e)); % vfoptions.lowmemory>0
end
pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension as no_z

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end

    end

    Vtilde=V;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V for naive
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    Vtilde=V;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=sum(V_Jplus1.*pi_e_J(1,:,N_j),2); % Note: The V_Jplus1 input should be V for naive

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
        % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
        % First V
        entireRHS=ReturnMatrix+beta*EV; % *repmat(EV,1,N_a,N_e); % Use the two-future-periods discount factor
        [Vtemp,~]=max(entireRHS,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        % Now Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*EV; %*repmat(EV,1,N_a,N_e); % Use today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
        Policy(:,:,N_j)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);

            % For naive, we compute V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vtilde(:,e_c,N_j)=shiftdim(Vtemp,1); % Evaluate what would have done under quasi-hyperbolic discounting
            Policy(:,e_c,N_j)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        end

    end
end



%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=V(:,:,jj+1); % Use V (goes into the equation to determine V)
    EV=sum(EV.*pi_e_J(1,:,jj),2);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
        % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
        % First V
        entireRHS=ReturnMatrix+beta*EV; %*repmat(EV,1,N_a,N_e); % Use the two-future-periods discount factor
        [Vtemp,~]=max(entireRHS,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        % Now Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*EV; %*repmat(EV,1,N_a,N_e); % Use today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
        Policy(:,:,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);

            % For naive, we compute V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS_e,[],1);
            V(:,e_c,jj)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vtilde(:,e_c,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under quasi-hyperbolic discounting
            Policy(:,e_c,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        end

    end
end

end
