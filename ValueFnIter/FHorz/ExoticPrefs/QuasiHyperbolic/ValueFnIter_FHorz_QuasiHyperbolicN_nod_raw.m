function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end

    Vtilde=V;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V for naive
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    Vtilde=V;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    EV=V_Jplus1; % Note: The V_Jplus1 input should be V for naive

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        % Use sparse for a few lines until sum over zprime
        EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
        % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
        % First V
        entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        [Vtemp,~]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp;
        % Now Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1); % Use today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,N_j)=Vtemp; % Evaluate what would have done under exponential discounting
        Policy(:,:,N_j)=maxindex; % Use the policy from solving the problem of Vtilde

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=EV.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            % For naive, we compute V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS_z=ReturnMatrix_z+beta*EV_z; %*EV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            % Now Vtilde and Policy
            entireRHS_z=ReturnMatrix_z+beta0beta*EV_z; %*EV_z.*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vtilde(:,z_c,N_j)=Vtemp; % Evaluate what would have done under exponential discounting
            Policy(:,z_c,N_j)=maxindex; % Use the policy from solving the problem of Vtilde
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

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % Use sparse for a few lines until sum over zprime
        EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
        % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
        % First V
        entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        [Vtemp,~]=max(entireRHS,[],1);
        V(:,:,jj)=Vtemp;
        % Now Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1); % Use today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,jj)=Vtemp; % Evaluate what would have done under exponential discounting
        Policy(:,:,jj)=maxindex; % Use the policy from solving the problem of Vtilde

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=EV.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            % For naive, we compute V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS_z=ReturnMatrix_z+beta*EV_z; %*EV_z.*ones(1,N_a,1); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            % Now Vtilde and Policy
            entireRHS_z=ReturnMatrix_z+beta0beta*EV_z; %*EV_z.*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vtilde(:,z_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
            Policy(:,z_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
        end
    end
end

end
