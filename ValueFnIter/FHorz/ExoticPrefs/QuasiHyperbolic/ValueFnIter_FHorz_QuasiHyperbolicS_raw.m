function [Vunderbar,Policy2,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end

    Vunderbar=Vhat;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be Vunderbar for sophisticated
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    Vunderbar=Vhat;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=V_Jplus1; % Note: The V_Jplus1 input should be Vunderbar for sophisticated

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Use sparse for a few lines until sum over zprime
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(EV,ones(N_d,1));

        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,:,N_j)=Vtemp; % Note that this is Vhat when sophisticated
        Policy(:,:,N_j)=maxindex; % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1);
        Vunderbar(:,:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            entireEV_z=kron(EV_z,ones(N_d,1));

            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp; % Note that this is Vhat when sophisticated
            Policy(:,z_c,N_j)=maxindex; % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS_z=ReturnMatrix_z+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
            Vunderbar(:,z_c,N_j)=entireRHS_z(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
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

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);

        % Use sparse for a few lines until sum over zprime
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(EV,N_d,1,1);

        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*entireEV; %*entireEV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vhat(:,:,jj)=Vtemp; % Note that this is Vhat when sophisticated
        Policy(:,:,jj)=maxindex; % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*entireEV; %*entireEV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1)+shiftdim(N_d*N_a*N_a*(0:1:N_z-1),-1);
        Vunderbar(:,:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            entireEV_z=kron(EV_z,ones(N_d,1));

            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vhat(:,z_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
            Policy(:,z_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS_z=ReturnMatrix_z+beta*entireEV_z; %*entireEV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_d*N_a*(0:1:N_a-1);
            Vunderbar(:,z_c,jj)=entireRHS_z(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); % NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
