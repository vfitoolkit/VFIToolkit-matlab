function [Vtilde,Policy3,Valt,Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% semiz is semi-exog state
% Naive quasi-hyperbolic dual.  Two independent maximisations over the same argmax axis:
%   Valt/Policy3alt maximise  F + beta*EV        (the exponential value; drives the recursion)
%   Vtilde/Policy3  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0 is received as a trailing input.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);

Valt=zeros(N_a,N_semiz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,a1prime seperately
Policy3alt=zeros(4,N_a,N_semiz,N_j,'gpuArray');
Policy3=zeros(4,N_a,N_semiz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

% For the return function we just want (I'm just guessing that as I need them N_j times it will be fractionally faster to put them together now)
n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_z]; Level=0, Refine=0
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy3alt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3alt(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy3alt(3,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy3alt(4,:,:,N_j)=ceil(maxindex/N_d); % d4


    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_semiz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy3alt(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3alt(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3alt(3,:,z_c,N_j)=ceil(d_ind/N_d12);
            Policy3alt(4,:,z_c,N_j)=ceil(maxindex/N_d);
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy3(:,:,:,N_j)=Policy3alt(:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]

    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EVinf=(V_Jplus1==-Inf);
            EV=V_Jplus1;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_semiz_d3';
            EV(EVinf*(pi_semiz_d3'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_semiz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_semiz); % [N_d2*N_a1,N_a2,N_semiz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d2,a1prime, a2,z)

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);

        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_semiz_d3(z_c,:));
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3z=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3z(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z); % probability of lower grid point+ probability of upper grid point
                entireEV_z(aprimeProbs_d3z==0)=EV2(aprimeProbs_d3z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV_z(aprimeProbs_d3z==1)=EV1(aprimeProbs_d3z==1);
                % entireEV_z is (d2,a1prime, a2)

                entireRHS_alt=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,N_a1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp_alt;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex_alt;
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,N_a1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp_tilde;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex_tilde;
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d2
    Valt(:,:,N_j)=V_jj;
    Policy3alt(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3alt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3alt(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy3alt(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d2
    Vtilde(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime

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
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_semiz_d3';
            EV(EVinf*(pi_semiz_d3'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_semiz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_semiz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_semiz); % [N_d2*N_a1,N_a2,N_semiz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d2,a1prime, a2,z)

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            % d3_val=d3_grid(d3_c);
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=EVpre.*(ones(N_a,1,'gpuArray')*pi_semiz_d3(z_c,:));
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2);

                % Switch EV_z from being in terms of aprime to being in terms of d and a
                EV1=reshape(EV_z(aprimeIndex),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the lower aprime
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2*N_a1,N_a2]); % (d2,a1prime,a2), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3z=repmat(a2primeProbs,N_a1,1); % [N_d2*N_a1,N_a2]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3z(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z); % probability of lower grid point+ probability of upper grid point
                entireEV_z(aprimeProbs_d3z==0)=EV2(aprimeProbs_d3z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV_z(aprimeProbs_d3z==1)=EV1(aprimeProbs_d3z==1);
                % entireEV_z is (d,a1prime, a2)

                entireRHS_alt=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,N_a1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,z_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,N_a1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,z_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end

    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,jj)=V_jj;
    Policy3alt(3,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3alt(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3alt(2,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy3alt(4,:,:,jj)=ceil(d12a1prime_ind/N_d12); % a1prime

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,jj)=ceil(d12a1prime_ind/N_d12); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
