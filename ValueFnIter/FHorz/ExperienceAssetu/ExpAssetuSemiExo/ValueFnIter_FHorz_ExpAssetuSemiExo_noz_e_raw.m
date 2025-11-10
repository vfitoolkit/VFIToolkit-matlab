function [V,Policy4]=ValueFnIter_FHorz_ExpAssetuSemiExo_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,n_u,N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
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
N_u=prod(n_u);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,a1prime seperately
Policy4=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

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

        ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz,N_e]
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy4(3,:,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy4(4,:,:,:,N_j)=ceil(maxindex/N_d); % d4

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz]
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy4(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy4(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1); % d2
            Policy4(3,:,:,e_c,N_j)=ceil(d_ind/N_d12); % d3
            Policy4(4,:,:,e_c,N_j)=ceil(maxindex/N_d); % d4
        end

    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d)+1;
                d12_ind=rem(d_ind-1,N_d12)+1;
                Policy4(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4(3,:,z_c,e_c,N_j)=ceil(d_ind/N_d12);
                Policy4(4,:,z_c,e_c,N_j)=ceil(maxindex/N_d);
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]
    
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
            % (d,aprime,a,z)

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)
            
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
                % (d,aprime,a,z)

                entireRHS_d3e=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
        
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                EV_z=EV(:,:,z_c);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_a1,n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0);

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountFactorParamsVec*repelem(EV_z,N_d1,N_a1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d2
    V(:,:,:,N_j)=V_jj;
    Policy4(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy4(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_semiz);  % [N_d2*N_a1,N_a2,N_u,N_semiz]
    
    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            ReturnMatrix_d3=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);
            % (d,aprime,a,z)

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0);
                % (d,aprime,a,z)

                entireRHSe=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHSe,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_u,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs); % probability of lower grid point+ probability of upper grid point
            % Already applied the probabilities from interpolating onto grid
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a1prime,a2,semiz)

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                EV_z=EV(:,:,z_c);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0);

                    entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(EV_z,N_d1,N_a1);

                    % Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);

                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy4(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4(2,:,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy4(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
