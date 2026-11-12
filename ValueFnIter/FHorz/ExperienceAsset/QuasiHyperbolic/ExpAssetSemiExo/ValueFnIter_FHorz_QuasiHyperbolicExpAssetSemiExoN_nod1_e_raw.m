function [Vtilde,Policy3,Valt,Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state
% Naive quasi-hyperbolic dual.  Two independent maximisations over the same argmax axis:
%   Valt/Policy3alt maximise  F + beta*EV        (the exponential value; drives the recursion)
%   Vtilde/Policy3  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0 is received as a trailing input.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy3alt=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
V_ford3_alt=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz,N_e]; Level=0, Refine=0
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3alt(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy3alt(2,:,:,:,N_j)=ceil(d_ind/N_d2);
        Policy3alt(3,:,:,:,N_j)=ceil(maxindex/N_d23);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz]; Level=0, Refine=0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
            Policy3alt(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3alt(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
            Policy3alt(3,:,:,e_c,N_j)=ceil(maxindex/N_d23);
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:N_semiz);
            z_val=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,semizblock,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy3alt(1,:,semizblock,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy3alt(2,:,semizblock,e_c,N_j)=ceil(d_ind/N_d2);
                Policy3alt(3,:,semizblock,e_c,N_j)=ceil(maxindex/N_d23);
            end
        end

    elseif vfoptions.lowmemory==3

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy3alt(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy3alt(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
                Policy3alt(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d23);
            end
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy3(:,:,:,:,N_j)=Policy3alt(:,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]

    % Using V_Jplus1
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d2,a1prime, a2,z)

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(EV,1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(EV,1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d2,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                % (d,aprime,a,z)

                entireRHS_alt=ReturnMatrix_d3e+DiscountedEV_alt;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+DiscountedEV_tilde;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,semizblock);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,semizblock);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_alt=ReturnMatrix_d3z+DiscountedEV_z_alt;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3z+DiscountedEV_z_tilde;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,N_j);
                    DiscountedEV_z_alt=DiscountedEV_alt(:,:,z_c);
                    DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,z_c);

                    ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_alt=ReturnMatrix_d3z+DiscountedEV_z_alt;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=Vtemp_alt;
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=maxindex_alt;
                    entireRHS_tilde=ReturnMatrix_d3z+DiscountedEV_z_tilde;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=Vtemp_tilde;
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=maxindex_tilde;
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d2
    Valt(:,:,:,N_j)=V_jj;
    Policy3alt(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3alt(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3alt(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d2
    Vtilde(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime


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
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(EV,1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(EV,1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                % (d,aprime,a,z)

                entireRHS_alt=ReturnMatrix_d3e+DiscountedEV_alt;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+DiscountedEV_tilde;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,semizblock);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,semizblock);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_alt=ReturnMatrix_d3ze+DiscountedEV_z_alt;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+DiscountedEV_z_tilde;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            EVinf=(EVpre==-Inf);
            EV=EVpre;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=EV*pi_bothz';
            EV(EVinf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=reshape(EV,[N_a,1,N_bothz]); % contract over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

            % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
            aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

            % Apply the aprimeProbs
            EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
            EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
            EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            % entireEV is (d,a1prime, a2,z)

            EVbase_qh=repelem(EV,1,N_a1,1);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,jj);
                    DiscountedEV_z_alt=DiscountedEV_alt(:,:,z_c);
                    DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,z_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_alt=ReturnMatrix_d3ze+DiscountedEV_z_alt;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+DiscountedEV_z_tilde;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end

    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,jj)=V_jj;
    Policy3alt(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3alt(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3alt(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime


end


%% For experience asset, just output Policy as is and then use Case2 to UnKron

end
