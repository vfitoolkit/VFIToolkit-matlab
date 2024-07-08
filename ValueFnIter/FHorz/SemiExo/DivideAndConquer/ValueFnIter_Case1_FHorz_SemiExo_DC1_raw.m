function [V,Policy3]=ValueFnIter_Case1_FHorz_SemiExo_DC1_raw(n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
V_ford2_jjtemp=zeros(N_a,N_semiz*N_z,'gpuArray'); % conditional on d2
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;
Policytemp=zeros(N_a,N_semiz*N_z,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);


if ~isfield(vfoptions,'V_Jplus1')

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,n_d2], n_bothz, [d1_grid; d2_grid], a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z)
        [RMtemp_ii,maxindex1]=max(ReturnMatrix_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2(:)+N_d*repmat((0:1:vfoptions.level1n-1)',N_bothz,1)+N_d*vfoptions.level1n*repelem((0:1:N_bothz-1)',vfoptions.level1n,1)); % aprime

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,2);
        Policytemp(level1ii,:)=maxindex2+N_d*(reshape(maxindex1d,[vfoptions.level1n,N_bothz])-1); % d,aprime

        % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
        [maxaprimeii,~]=max(max(maxindex1,[],1),[],4); % Get the max aprime index, conditional on a, across all possible d & z
        [minaprimeii,~]=min(min(maxindex1,[],1),[],4); % Get the min aprime index, conditional on a, across all possible d & z
        maxaprimeii=maxaprimeii(:); % turn into column vector
        minaprimeii=minaprimeii(:); % turn into column vector
        % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.

        % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
        %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
        for ii=1:(vfoptions.level1n-1)
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,n_d2], n_bothz, [d1_grid; d2_grid], a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,N_j)=shiftdim(Vtempii,1);
            Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d*(minaprimeii(ii)-1);
        end

        % Deal with policy for semi-exo
        d_ind=shiftdim(rem(Policytemp-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(Policytemp/N_d),-1);

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    for d2_c=1:N_d2
        % Note: By definition V_Jplus1 does not depend on d (only aprime)
        pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

        EV=V_Jplus1.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repmat(shiftdim(EV,-1),N_d1,1,1,1); % [d1,aprime,1,z]

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z)
        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        % maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2(:)+N_d1*repmat((0:1:vfoptions.level1n-1)',N_bothz,1)+N_d1*vfoptions.level1n*repelem((0:1:N_bothz-1)',vfoptions.level1n,1)); % aprime

        % Store
        V_ford2_jjtemp(level1ii,:)=shiftdim(Vtempii,2);
        Policytemp(level1ii,:)=shiftdim(maxindex2,2)+N_d1*(reshape(maxindex1d,[vfoptions.level1n,N_bothz])-1); % d,aprime

        % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
        [maxaprimeii,~]=max(max(maxindex1,[],1),[],4); % Get the max aprime index, conditional on a, across all possible d & z
        [minaprimeii,~]=min(min(maxindex1,[],1),[],4); % Get the min aprime index, conditional on a, across all possible d & z
        maxaprimeii=maxaprimeii(:); % turn into column vector
        minaprimeii=minaprimeii(:); % turn into column vector
        % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.
    
        % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
        %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
        for ii=1:(vfoptions.level1n-1)
            if maxaprimeii(ii+1)>minaprimeii(ii)
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=(repmat(1:1:N_d1,1,maxaprimeii(ii+1)-minaprimeii(ii)+1)+N_d1*repelem(minaprimeii(ii)-1:1:maxaprimeii(ii+1)-1,1,N_d1))'+N_d1*N_a*(0:1:N_bothz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxaprimeii(ii+1)-minaprimeii(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jjtemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
                Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            else
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid(minaprimeii(ii)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                daprimez=((1:1:N_d1)+N_d1*(minaprimeii(ii)-1))'+N_d1*N_a*(0:1:N_bothz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jjtemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
                Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            end
        end

        V_ford2_jj(:,:,d2_c)=V_ford2_jjtemp;
        Policy_ford2_jj(:,:,d2_c)=Policytemp;

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=V(:,:,jj+1);

    for d2_c=1:N_d2
        % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
        pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order

        EV=VKronNext_j.*shiftdim(pi_bothz',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repmat(shiftdim(EV,-1),N_d1,1,1,1); % [d1,aprime,1,z]

        % n-Monotonicity
        ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z)
        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        % maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2(:)+N_d1*repmat((0:1:vfoptions.level1n-1)',N_bothz,1)+N_d1*vfoptions.level1n*repelem((0:1:N_bothz-1)',vfoptions.level1n,1)); % aprime

        % Store
        V_ford2_jjtemp(level1ii,:)=shiftdim(Vtempii,2);
        Policytemp(level1ii,:)=shiftdim(maxindex2,2)+N_d1*(reshape(maxindex1d,[vfoptions.level1n,N_bothz])-1); % d,aprime

        % We just want aprime to use in n-Monotonicity. But we want it conditional on a, but allow for full range that different d give
        [maxaprimeii,~]=max(max(maxindex1,[],1),[],4); % Get the max aprime index, conditional on a, across all possible d & z
        [minaprimeii,~]=min(min(maxindex1,[],1),[],4); % Get the min aprime index, conditional on a, across all possible d & z
        maxaprimeii=maxaprimeii(:); % turn into column vector
        minaprimeii=minaprimeii(:); % turn into column vector
        % COMMENT TO SELF: I tried to instead "Get max aprime index, conditional on (a,z), across all possible d". But then I have to add a for-loop over z (when doing ReturnMatrix_ii) and this was slower.
    
        % Note: because of monotonicity, can just use minaprimeii(ii) & maxaprimeii(ii+1)
        %       [as minaprimeii(ii) will be less than or equal to minaprime(ii+1) anyway, and analagously for max]
        for ii=1:(vfoptions.level1n-1)
            if maxaprimeii(ii+1)>minaprimeii(ii)
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid(minaprimeii(ii):maxaprimeii(ii+1)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=(repmat(1:1:N_d1,1,maxaprimeii(ii+1)-minaprimeii(ii)+1)+N_d1*repelem(minaprimeii(ii)-1:1:maxaprimeii(ii+1)-1,1,N_d1))'+N_d1*N_a*(0:1:N_bothz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxaprimeii(ii+1)-minaprimeii(ii)+1),1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jjtemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
                Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            else
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, [n_d1,special_n_d2], n_bothz, [d1_grid; d2_grid(d2_c)], a_grid(minaprimeii(ii)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                daprimez=((1:1:N_d1)+N_d1*(minaprimeii(ii)-1))'+N_d1*N_a*(0:1:N_bothz-1); % all the d, with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,1,N_bothz]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V_ford2_jjtemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
                Policytemp(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex,1)+N_d1*(minaprimeii(ii)-1);
            end
        end

        V_ford2_jj(:,:,d2_c)=V_ford2_jjtemp;
        Policy_ford2_jj(:,:,d2_c)=Policytemp;
    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end