function varargout=ValueFnIter_Case1_FHorz_QuasiHyperbolic_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J,pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z

%%
special_n_e=ones(1,length(n_e)); % vfoptions.lowmemory>0
pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
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
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

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
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
            
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compue V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS,[],1);
            V(:,:,:,N_j)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e);
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde(:,:,:,N_j)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
            Policy(:,:,:,N_j)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,:,N_j)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
            Policy(:,:,:,N_j)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a*N_a*N_z*(0:1:N_e-1),-2);
            Vunderbar(:,:,:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
                       
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor                
                [Vtemp,~]=max(entireRHS_e,[],1);
                V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
                % Now Vtilde and Policy
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vtilde(:,:,e_c,N_j)=shiftdim(Vtemp,1); % Evaluate what would have done under quasi-hyperbolic discounting
                Policy(:,:,e_c,N_j)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')  
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat) 
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V(:,:,e_c,N_j)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
                Policy(:,:,e_c,N_j)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
                maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1);
                Vunderbar(:,:,e_c,N_j)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
        
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireEV_z=kron(EV_z,ones(N_d,1));
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,N_j)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    Vtilde(:,z_c,e_c,N_j)=Vtemp; % Evaluate what would have done under quasi-hyperbolic discounting
                    Policy(:,z_c,e_c,N_j)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,N_j)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,e_c,N_j)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
                    maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
                    Vunderbar(:,z_c,e_c,N_j)=entireRHS_ez(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
            
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
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.
    
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        VKronNext_j=V(:,:,:,jj+1); % Use V (goes into the equation to determine V)
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        VKronNext_j=Vunderbar(:,:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
    end
            
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
            
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compue V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS,[],1);
            V(:,:,:,jj)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e);
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde(:,:,:,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
            Policy(:,:,:,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,:,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
            Policy(:,:,:,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*entireEV; %*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a*N_a*N_z*(0:1:N_e-1),-2);
            Vunderbar(:,:,:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
                       
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor                
                [Vtemp,~]=max(entireRHS_e,[],1);
                V(:,:,e_c,jj)=shiftdim(Vtemp,1);
                % Now Vtilde and Policy
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vtilde(:,:,e_c,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under quasi-hyperbolic discounting
                Policy(:,:,e_c,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')  
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat) 
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V(:,:,e_c,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
                Policy(:,:,e_c,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_e=ReturnMatrix_e+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
                maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1);
                Vunderbar(:,:,e_c,jj)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
        
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireEV_z=kron(EV_z,ones(N_d,1));
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    Vtilde(:,z_c,e_c,jj)=Vtemp; % Evaluate what would have done under quasi-hyperbolic discounting
                    Policy(:,z_c,e_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,e_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
                    maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
                    Vunderbar(:,z_c,e_c,jj)=entireRHS_ez(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
            
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

% The basic version just returns two outputs, but it is possible to request
% three as might want to see the 'other' value fn which is used in the expectations.
nOutputs = nargout;
if nOutputs==2
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        varargout={Vtilde,Policy2}; % Policy will be Policytilde, value fn is Vtilde
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        varargout={V,Policy2}; % Policy will be Policyhat, value fn is Vhat
    end
elseif nOutputs==3
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        varargout={Vtilde,Policy2,V}; % Policy will be Policytilde, value fns are Vtilde and V
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        varargout={V,Policy2,Vunderbar}; % Policy will be Policyhat, value fns are Vhat and Vunderbar
    end
end


end