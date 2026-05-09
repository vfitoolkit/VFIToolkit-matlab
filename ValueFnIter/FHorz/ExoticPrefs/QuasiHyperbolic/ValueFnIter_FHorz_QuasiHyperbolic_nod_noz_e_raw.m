function [V1,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolic_nod_noz_e_raw(n_a,n_e,N_j, a_grid,e_gridvals_J,pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Interpretation of output differs by Naive/Sophisticated
% Naive:         {Vtilde, Policy, V}
% Sophisticated: {Vhat,  Policy, Vunderbar}
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount.gives the name of the beta_0 is the additional discount factor parameter
%
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);
N_e=prod(n_e);

% Policy_extra=zeros(N_a,N_e,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

V=zeros(N_a,N_e,N_j,'gpuArray'); % If Naive, then this is V, if Sophisticated then this is Vhat.
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%
special_n_e=ones(1,length(n_e)); % vfoptions.lowmemory>0
pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end

    end

    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        Vtilde=V;
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        Vunderbar=V;
    end
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V if naive, Vunderbar if sophisticated
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    % Preallocate Vtilde and Vunderbar
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        Vtilde=V;
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        Vunderbar=V;
    end

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.

    VKronNext_j=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3); % Note: The V_Jplus1 input should be V if naive, Vunderbar if sophisticated
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        EV=VKronNext_j;
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
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
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*EV; %repmat(EV,1,N_a,N_e);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,N_j)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
            Policy(:,:,N_j)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*repmat(EV,1,N_a,N_e); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_e-1),-1);
            Vunderbar(:,:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j;
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
                       
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
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
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')  
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat) 
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V(:,e_c,N_j)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
                Policy(:,e_c,N_j)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1); % Use the two-future-periods discount factor
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                Vunderbar(:,e_c,N_j)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
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
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.

    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        VKronNext_j=V(:,:,:,jj+1); % Use V (goes into the equation to determine V)
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        VKronNext_j=Vunderbar(:,:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
    end
    
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec);

        EV=VKronNext_j;
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
            % First V
            entireRHS=ReturnMatrix+beta*EV; %repmat(EV,1,N_a,N_e); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS,[],1);
            V(:,:,jj)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS=ReturnMatrix+beta0beta*EV;%*repmat(EV,1,N_a,N_e); % Use today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde(:,:,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
            Policy(:,:,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*EV; %repmat(EV,1,N_a,N_e);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
            Policy(:,:,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*EV; %*repmat(EV,1,N_a,N_e); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_a*(0:1:N_a-1)+shiftdim(N_a*N_a*(0:1:N_z-1),-1);
            Vunderbar(:,:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j;
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
                       
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
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
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')  
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat) 
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS_e=ReturnMatrix_e+beta0beta*EV; %.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V(:,e_c,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
                Policy(:,e_c,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_e=ReturnMatrix_e+beta*EV; %.*ones(1,N_a,1); % Use the two-future-periods discount factor
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                Vunderbar(:,e_c,jj)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
        
    end
end

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    V1=V;
    Valt=Vtilde;
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    V1=Vhat;
    Valt=Vunderbar;
end


end
