function [Vunderbar,Policy2,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J,pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Vhat=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e)); % vfoptions.lowmemory>0
    pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end

    end

    Vunderbar=Vhat;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be Vunderbar for sophisticated
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    Vunderbar=Vhat;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3); % Note: The V_Jplus1 input should be Vunderbar for sophisticated

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        % (d,aprime,a,e)

        entireEV=repelem(VKronNext_j,N_d,1,1);

        % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,N_e);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
        Policy(:,:,N_j)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,N_e); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_e-1),-1);
        Vunderbar(:,:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

    elseif vfoptions.lowmemory==1
        entireEV=repelem(VKronNext_j,N_d,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);

            % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vhat(:,e_c,N_j)=Vtemp; % Note that this is Vhat when sophisticated
            Policy(:,e_c,N_j)=maxindex; % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
            Vunderbar(:,e_c,N_j)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

        end

    end
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

    VKronNext_j=Vunderbar(:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)

    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        % (d,aprime,a,e)

        entireEV=repelem(VKronNext_j,N_d,1,1);

        % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,N_e);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,:,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
        Policy(:,:,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,N_e); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_e-1),-1);
        Vunderbar(:,:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

    elseif vfoptions.lowmemory==1
        entireEV=repelem(VKronNext_j,N_d,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);

            % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            Vhat(:,e_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
            Policy(:,e_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
            Vunderbar(:,e_c,jj)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

        end

    end
end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
