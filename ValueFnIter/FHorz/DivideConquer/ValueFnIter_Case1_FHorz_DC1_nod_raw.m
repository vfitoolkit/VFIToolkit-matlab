function [V, Policy]=ValueFnIter_Case1_FHorz_DC1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;
Policytemp=zeros(N_a,N_z,'gpuArray');

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec);

    % size(ReturnMatrix_ii) % (aprime, a,z)
    % [n_a,vfoptions.level1n,n_z]

    %Calc the max and it's index
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);

    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policytemp(level1ii,:)=shiftdim(maxindex,1);

    % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
    [maxaprimeii,~]=max(maxindex,[],3); % Get the max aprime index, conditional on a, across all possible z
    [minaprimeii,~]=min(maxindex,[],3); % Get the min aprime index, conditional on a, across all possible z
    maxaprimeii=maxaprimeii(:); % turn into column vector
    minaprimeii=minaprimeii(:); % turn into column vector
    % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.

    % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
    %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
    for ii=1:(vfoptions.level1n-1)
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:,N_j)=shiftdim(Vtempii,1);
        Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+(minaprimeii(ii)-1);
    end

    Policy(:,:,N_j)=Policytemp;
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtempii,maxindex]=max(entireRHS_ii,[],1);

    V(level1ii,:,N_j)=shiftdim(Vtempii,1);
    Policytemp(level1ii,:)=shiftdim(maxindex,1);

    % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
    [maxaprimeii,~]=max(maxindex,[],3); % Get the max aprime index, conditional on a, across all possible d & z
    [minaprimeii,~]=min(maxindex,[],3); % Get the min aprime index, conditional on a, across all possible d & z
    maxaprimeii=maxaprimeii(:); % turn into column vector
    minaprimeii=minaprimeii(:); % turn into column vector
    % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.
    

    % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
    %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
    for ii=1:(vfoptions.level1n-1)
        if maxaprimeii(ii+1)>minaprimeii(ii)
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            aprimez=(minaprimeii(ii):1:maxaprimeii(ii+1))'+N_a*(0:1:N_z-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxaprimeii(ii+1)-minaprimeii(ii)+1),1,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d*(minaprimeii(ii)-1);
        else
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(minaprimeii(ii)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            aprimez=minaprimeii(ii)+N_a*(0:1:N_z-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,1,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+(minaprimeii(ii)-1);
        end
    end

    Policy(:,:,N_j)=Policytemp;
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,jj+1);
    
    % Use sparse for a few lines until sum over zprime
    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
            
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec);

    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

    %Calc the max and it's index
    [Vtempii,maxindex]=max(entireRHS_ii,[],1);

    V(level1ii,:,jj)=shiftdim(Vtempii,1);
    Policytemp(level1ii,:)=shiftdim(maxindex,1);

    % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
    [maxaprimeii,~]=max(maxindex,[],3); % Get the max aprime index, conditional on a, across all possible d & z
    [minaprimeii,~]=min(maxindex,[],3); % Get the min aprime index, conditional on a, across all possible d & z
    maxaprimeii=maxaprimeii(:); % turn into column vector
    minaprimeii=minaprimeii(:); % turn into column vector
    % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.
    
    % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
    %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
    for ii=1:(vfoptions.level1n-1)
        if maxaprimeii(ii+1)>minaprimeii(ii)
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec);
            aprimez=(minaprimeii(ii):1:maxaprimeii(ii+1))'+N_a*(0:1:N_z-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[(maxaprimeii(ii+1)-minaprimeii(ii)+1),1,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+(minaprimeii(ii)-1);
        else
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(minaprimeii(ii)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec);
            aprimez=minaprimeii(ii)+N_a*(0:1:N_z-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV(aprimez(:)),[1,1,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,jj)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+(minaprimeii(ii)-1);
        end
    end

    Policy(:,:,jj)=Policytemp;
end


end