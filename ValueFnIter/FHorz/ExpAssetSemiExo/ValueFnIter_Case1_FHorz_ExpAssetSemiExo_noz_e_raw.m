function [V,Policy3]=ValueFnIter_Case1_FHorz_ExpAssetSemiExo_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% semiz is semi-exog state

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,a1prime seperately
Policy3=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
d3_grid=gpuArray(d3_grid);
a1_grid=gpuArray(a1_grid);
a2_grid=gpuArray(a2_grid);

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d_grid=[d1_grid; d2_grid; d3_grid];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_semiz, n_e, d_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % [N_d*N_a1,N_a1*N_a2,N_z]
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy3(3,:,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy3(4,:,:,:,N_j)=ceil(maxindex/N_d); % d4

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, n_semiz, special_n_e, d_grid, a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec); % [N_d*N_a1,N_a1*N_a2,N_z]
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy3(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy3(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1); % d2
            Policy3(3,:,:,e_c,N_j)=ceil(d_ind/N_d12); % d3
            Policy3(4,:,:,e_c,N_j)=ceil(maxindex/N_d); % d4
        end

    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d, n_a1,n_a2, special_n_semiz,special_n_e, d_grid, a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec);
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d)+1;
                d12_ind=rem(d_ind-1,N_d12)+1;
                Policy3(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy3(3,:,z_c,e_c,N_j)=ceil(d_ind/N_d12);
                Policy3(4,:,z_c,e_c,N_j)=ceil(maxindex/N_d);
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_semiz]
    end

    % Using V_Jplus1
    V_Jplus1=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, n_semiz,n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            % (d,aprime,a,z)

            EV=V_Jplus1.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d2,a1prime, a2,z)

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);
            
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, n_semiz,special_n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
                % (d,aprime,a,z)

                EV=V_Jplus1.*shiftdim(pi_semi_d3',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                % Switch EV from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV is (d2,a1prime, a2,z)

                entireRHS_d3e=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
        
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, special_n_semiz, special_n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_semi_d3(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    % Switch EV_z from being in terms of aprime to being in terms of d and a
                    EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                    skipinterp=(EV1==EV2);
                    aprimeProbs(skipinterp)=0; % effectively skips interpolation

                    % Apply the aprimeProbs
                    entireEV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                    % entireEV_z is (d2,a1prime, a2)

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d2
    V(:,:,:,N_j)=V_jj;
    Policy3(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    if vfoptions.lowmemory>0 || vfoptions.paroverz==0
        aprimeProbs=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
    else % lowmemory=0 and paroverz=1
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_semiz]
    end

    VKronNext_j=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, n_semiz,n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
            % (d,aprime,a,z)

            EV=VKronNext_j.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % entireEV is (d2,a1prime, a2,z)

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3            
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, n_semiz,special_n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
                % (d,aprime,a,z)

                EV=VKronNext_j.*shiftdim(pi_semi_d3',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                % Switch EV from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                % entireEV is (d2,a1prime, a2,z)

                entireRHSe=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHSe,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d3_val=d3_grid(d3_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,jj);
                    ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, [n_d1,n_d2,1], n_a1,n_a2, special_n_semiz,special_n_e, [d1_grid;d2_grid;d3_val], a1_grid, a2_grid, z_val,e_val, ReturnFnParamsVec);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_semi_d3(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    % Switch EV_z from being in terms of aprime to being in terms of d and a
                    EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                    skipinterp=(EV1==EV2);
                    aprimeProbs(skipinterp)=0; % effectively skips interpolation

                    % Apply the aprimeProbs
                    entireEV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
                    % entireEV_z is (d,a1prime, a2)

                    entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);

                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy3(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12); % a1prime
end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end