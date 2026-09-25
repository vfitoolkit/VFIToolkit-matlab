function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_GI2A_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, u_gridvals, pi_z_J, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAssetuSemiExo_GI2A_e_raw.
% ExperienceAsset + semi-exogenous shocks, with the grid interpolation layer on a1. GPU only.
%
% Naive:  Valt_j   = max_{d,a1'} u + beta*E[Valt_{j+1}]         (exponential discounter)
%         Vtilde_j = max_{d,a1'} u + beta_0*beta*E[Valt_{j+1}]  (agent's perceived choice)
% The two discount factors generally pick different GI midpoints, so each pass re-derives its
% own midpoint and its own layer-2 return matrix, and each keeps its own d3 choice, its own
% (midpoint, L2 index) policy pair and its own L2 flag.
% beta0 is received as a trailing input.
% SemiExo graft of ValueFnIter_FHorz_ExpAsset_GI2A_e_raw.
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% Policy stores (d1, d2, d3, joint(a1prime,a2prime), a1primeL2ind); the 4th row is a1prime+N_a1*(a2prime-1), a1prime being the lower grid point.
% The grid interpolation (a1) keeps the source GI2A math: midpoint, a1primeindexes, fine-grid interpolation.
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime,a2prime),a1primeL2ind seperately
Policy=zeros(6,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy(6,:,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1
Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policyalt=zeros(6,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
Policyalt(6,:,:,:,:)=2;

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1

% Preallocate (for the sections, which loop over d3)
V_ford3_tilde=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy4_ford3_tilde=zeros(4,N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
flag_ford3_tilde=2*ones(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
V_ford3_alt=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy4_ford3_alt=zeros(4,N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
flag_ford3_alt=2*ones(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    % Terminal period: no continuation, so the exponential and the QH-perceived problems coincide
    Valt(:,:,:,N_j)=Vtilde(:,:,:,N_j);
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j);
    Policyalt(6,:,:,:,N_j)=Policy(6,:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV_aprime=EV_aprime.*pi_u; EV_aprime(isnan(EV_aprime))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV_aprime=squeeze(sum(EV_aprime,3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            % alt pass (exponential discounter): F + beta*EV
            entireRHS=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % tilde pass (QH-perceived): F + beta0*beta*EV
            entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                % alt pass (exponential discounter): F + beta*EV
                entireRHS=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                % tilde pass (QH-perceived): F + beta0*beta*EV
                entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,zind);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    % alt pass (exponential discounter): F + beta*EV
                    entireRHS=ReturnMatrix+beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    % tilde pass (QH-perceived): F + beta0*beta*EV
                    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    % alt pass (exponential discounter): F + beta*EV
                    entireRHS=ReturnMatrix+beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    % tilde pass (QH-perceived): F + beta0*beta*EV
                    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,N_j)=V_jjalt;
    Policyalt(3,:,:,:,N_j)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,N_j)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]); % d1
    Policyalt(2,:,:,:,N_j)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]); % d2
    Policyalt(4,:,:,:,N_j)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policyalt(5,:,:,:,N_j)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policyalt(6,:,:,:,N_j)=reshape(flag_ford3_alt((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)),[1,N_a,N_bothz,N_e]);

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,N_j)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,N_j)=reshape(flag_ford3_tilde((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);
    EV_aprime=EV_aprime.*pi_u; EV_aprime(isnan(EV_aprime))=0; % a zero pi_u against an infinite node gives 0*(-Inf)=NaN, so zero the term BEFORE summing
    EV_aprime=squeeze(sum(EV_aprime,3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            % alt pass (exponential discounter): F + beta*EV
            entireRHS=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_alt(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_alt(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_alt(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_alt(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_alt(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            % tilde pass (QH-perceived): F + beta0*beta*EV
            entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind + N_d12*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_tilde(1,:,:,:,d3_c)=rem(d_ind-1,N_d1)+1; % d1
            Policy4_ford3_tilde(2,:,:,:,d3_c)=ceil(d_ind/N_d1); % d2
            Policy4_ford3_tilde(3,:,:,:,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4_ford3_tilde(4,:,:,:,d3_c)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind + N_d12*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_tilde(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                % alt pass (exponential discounter): F + beta*EV
                entireRHS=ReturnMatrix+beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_alt(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_alt(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_alt(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_alt(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_alt(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                % tilde pass (QH-perceived): F + beta0*beta*EV
                entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1,1);
                [~,maxindex]=max(entireRHS,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;
                Policy4_ford3_tilde(1,:,:,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                Policy4_ford3_tilde(2,:,:,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                Policy4_ford3_tilde(3,:,:,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                Policy4_ford3_tilde(4,:,:,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_tilde(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,zind);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    % alt pass (exponential discounter): F + beta*EV
                    entireRHS=ReturnMatrix+beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_alt(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    % tilde pass (QH-perceived): F + beta0*beta*EV
                    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;
                    Policy4_ford3_tilde(1,:,zind,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,zind,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,zind,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,zind,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,zind,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]); % undiscounted; beta/beta0beta applied at the use sites
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    % alt pass (exponential discounter): F + beta*EV
                    entireRHS=ReturnMatrix+beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_alt+beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_alt(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_alt(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_alt(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_alt(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_alt(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_alt(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    % tilde pass (QH-perceived): F + beta0*beta*EV
                    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV_z,N_d1,1,1,1,1,1,1);
                    [~,maxindex]=max(entireRHS,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_tilde+beta0beta*entireEVinterp_z(aprimez),[N_d12*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d12)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;

                    allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;
                    Policy4_ford3_tilde(1,:,z_c,e_c,d3_c)=rem(d_ind-1,N_d1)+1; % d1
                    Policy4_ford3_tilde(2,:,z_c,e_c,d3_c)=ceil(d_ind/N_d1); % d2
                    Policy4_ford3_tilde(3,:,z_c,e_c,d3_c)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
                    Policy4_ford3_tilde(4,:,z_c,e_c,d3_c)=maxindexL2a1; % a1primeL2ind

                    linidx_lower=d_ind                 + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d12*(n2long-1)+ N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_tilde(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_tilde(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jjalt,maxindexalt]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,jj)=V_jjalt;
    Policyalt(3,:,:,:,jj)=shiftdim(maxindexalt,-1); % d3 is just maxindexalt
    maxindexalt=reshape(maxindexalt,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    tempalt=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1) -1);
    Policyalt(1,:,:,:,jj)=reshape(Policy4_ford3_alt(1+tempalt),[1,N_a,N_bothz,N_e]); % d1
    Policyalt(2,:,:,:,jj)=reshape(Policy4_ford3_alt(2+tempalt),[1,N_a,N_bothz,N_e]); % d2
    Policyalt(4,:,:,:,jj)=reshape(Policy4_ford3_alt(3+tempalt),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policyalt(5,:,:,:,jj)=reshape(Policy4_ford3_alt(4+tempalt),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policyalt(6,:,:,:,jj)=reshape(flag_ford3_alt((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindexalt-1)),[1,N_a,N_bothz,N_e]);

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    temp=4*( (1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1) -1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_tilde(1+temp),[1,N_a,N_bothz,N_e]); % d1
    Policy(2,:,:,:,jj)=reshape(Policy4_ford3_tilde(2+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_tilde(3+temp),[1,N_a,N_bothz,N_e]); % joint(a1prime,a2prime)
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_tilde(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    Policy(6,:,:,:,jj)=reshape(flag_ford3_tilde((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
end


%% With grid interpolation, switch from midpoint to lower grid index
% Currently Policy(4,:) holds joint(a1prime midpoint,a2prime) and Policy(5,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch the a1prime part of the joint to 'lower grid point' and then have Policy(5,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(5,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust; % a1prime part of joint -> lower grid point
Policy(5,:,:,:,:)=Policy(5,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

adjustalt=(Policyalt(5,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(4,:,:,:,:)=Policyalt(4,:,:,:,:)-adjustalt; % a1prime part of joint -> lower grid point
Policyalt(5,:,:,:,:)=Policyalt(5,:,:,:,:)-(n2short+1)*(~adjustalt); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)


end
