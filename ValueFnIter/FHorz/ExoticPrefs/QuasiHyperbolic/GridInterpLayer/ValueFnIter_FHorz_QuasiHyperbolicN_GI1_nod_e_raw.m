function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_e_raw(n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI1_nod_e_raw.
% No d variables. Has z and e variables. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(2,N_a,N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
Policyalt=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_z,N_e,N_j,'gpuArray');

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1);

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

pi_e_J=shiftdim(pi_e_J,-2);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [~,maxindex]=max(ReturnMatrix,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

        isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,:,N_j)=shiftdim(maxindexL2,-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            [~,maxindex]=max(ReturnMatrix_e,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);

            isInfLower    = (ReturnMatrix_ii_e(1,     :,:) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_e(n2long,:,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(maxindexL2,-1);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                [~,maxindex]=max(ReturnMatrix_ze,[],1);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,special_n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);

                isInfLower    = (ReturnMatrix_ii_ze(1,     :) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_ze(n2long,:) == -Inf);
                inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
                inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint),-1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(maxindexL2,-1);
            end
        end
    end
    Vtilde=Valt;
    Policyalt(:,:,:,:,N_j)=Policy(:,:,:,:,N_j); % terminal: QH and exp discounter coincide
    PolicyL2flagalt(1,:,:,:,N_j)=PolicyL2flag(1,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j),3);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindexalt]=max(entireRHS,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimezalt=aprimeindexesalt+n2aprime*zind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,N_j)=shiftdim(Vtempii,1);

        isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        PolicyL2flagalt(1,:,:,:,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

        Policyalt(1,:,:,:,N_j)=shiftdim(squeeze(midpointalt),-1);
        Policyalt(2,:,:,:,N_j)=shiftdim(maxindexL2alt,-1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

        isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Vtilde(:,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,:,N_j)=shiftdim(maxindexL2,-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_e=ReturnMatrix_e+beta*EV;
            [~,maxindexalt]=max(entireRHS_e,[],1);
            midpointalt=max(min(maxindexalt,n_a-1),2);
            aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,special_n_e,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            aprimezalt=aprimeindexesalt+n2aprime*zind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,:,e_c,N_j)=shiftdim(Vtempii,1);

            isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagalt(1,:,:,e_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

            Policyalt(1,:,:,e_c,N_j)=shiftdim(squeeze(midpointalt),-1);
            Policyalt(2,:,:,e_c,N_j)=shiftdim(maxindexL2alt,-1);
            %% Vtilde (beta0*beta)
            entireRHS_e=ReturnMatrix_e+beta0beta*EV;
            [~,maxindex]=max(entireRHS_e,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Vtilde(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(maxindexL2,-1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHS_ze=ReturnMatrix_ze+beta*EV_z;
                [~,maxindexalt]=max(entireRHS_ze,[],1);
                midpointalt=max(min(maxindexalt,n_a-1),2);
                aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,special_n_z,special_n_e,aprime_grid(aprimeindexesalt),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp_z(aprimeindexesalt(:)),[n2long,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
                Valt(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);

                isInfLoweralt    = (ReturnMatrix_iialt(1,     :) == -Inf);
                isInfUpperalt    = (ReturnMatrix_iialt(n2long,:) == -Inf);
                inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
                inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
                PolicyL2flagalt(1,:,z_c,e_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

                Policyalt(1,:,z_c,e_c,N_j)=shiftdim(squeeze(midpointalt),-1);
                Policyalt(2,:,z_c,e_c,N_j)=shiftdim(maxindexL2alt,-1);
                %% Vtilde (beta0*beta)
                entireRHS_ze=ReturnMatrix_ze+beta0beta*EV_z;
                [~,maxindex]=max(entireRHS_ze,[],1);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,special_n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

                isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
                inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
                inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                Vtilde(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint),-1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(maxindexL2,-1);
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

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVsource=Valt(:,:,:,jj+1);
    EV=sum(EVsource.*pi_e_J(1,1,:,jj),3);
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, n_e, 0, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindexalt]=max(entireRHS,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimezalt=aprimeindexesalt+n2aprime*zind;
        entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
        Valt(:,:,:,jj)=shiftdim(Vtempii,1);

        isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        PolicyL2flagalt(1,:,:,:,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

        Policyalt(1,:,:,:,jj)=shiftdim(squeeze(midpointalt),-1);
        Policyalt(2,:,:,:,jj)=shiftdim(maxindexL2alt,-1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

        isInfLower    = (ReturnMatrix_ii(1,     :,:,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Vtilde(:,:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,:,jj)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,:,jj)=shiftdim(maxindexL2,-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, n_z, special_n_e, 0, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);
            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS_e=ReturnMatrix_e+beta*EV;
            [~,maxindexalt]=max(entireRHS_e,[],1);
            midpointalt=max(min(maxindexalt,n_a-1),2);
            aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,special_n_e,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            aprimezalt=aprimeindexesalt+n2aprime*zind;
            entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
            Valt(:,:,e_c,jj)=shiftdim(Vtempii,1);

            isInfLoweralt    = (ReturnMatrix_iialt(1,     :,:) == -Inf);
            isInfUpperalt    = (ReturnMatrix_iialt(n2long,:,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagalt(1,:,:,e_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

            Policyalt(1,:,:,e_c,jj)=shiftdim(squeeze(midpointalt),-1);
            Policyalt(2,:,:,e_c,jj)=shiftdim(maxindexL2alt,-1);
            %% Vtilde (beta0*beta)
            entireRHS_e=ReturnMatrix_e+beta0beta*EV;
            [~,maxindex]=max(entireRHS_e,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),e_val,ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*zind;
            entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

            isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Vtilde(:,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,:,e_c,jj)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,:,e_c,jj)=shiftdim(maxindexL2,-1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, 0, n_a, special_n_z, special_n_e, 0, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
                entireRHS_ze=ReturnMatrix_ze+beta*EV_z;
                [~,maxindexalt]=max(entireRHS_ze,[],1);
                midpointalt=max(min(maxindexalt,n_a-1),2);
                aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_iialt=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,special_n_z,special_n_e,aprime_grid(aprimeindexesalt),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_iialt=ReturnMatrix_iialt+beta*reshape(EVinterp_z(aprimeindexesalt(:)),[n2long,N_a]);
                [Vtempii,maxindexL2alt]=max(entireRHS_iialt,[],1);
                Valt(:,z_c,e_c,jj)=shiftdim(Vtempii,1);

                isInfLoweralt    = (ReturnMatrix_iialt(1,     :) == -Inf);
                isInfUpperalt    = (ReturnMatrix_iialt(n2long,:) == -Inf);
                inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
                inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
                PolicyL2flagalt(1,:,z_c,e_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

                Policyalt(1,:,z_c,e_c,jj)=shiftdim(squeeze(midpointalt),-1);
                Policyalt(2,:,z_c,e_c,jj)=shiftdim(maxindexL2alt,-1);
                %% Vtilde (beta0*beta)
                entireRHS_ze=ReturnMatrix_ze+beta0beta*EV_z;
                [~,maxindex]=max(entireRHS_ze,[],1);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn,special_n_z,special_n_e,aprime_grid(aprimeindexes),a_grid,z_val,e_val,ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);

                isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
                inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
                inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
                PolicyL2flag(1,:,z_c,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                Vtilde(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(1,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint),-1);
                Policy(2,:,z_c,e_c,jj)=shiftdim(maxindexL2,-1);
            end
        end
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:,:)<1+n2short+1);
Policy(1,:,:,:,:)=Policy(1,:,:,:,:)-adjust;
Policy(2,:,:,:,:)=adjust.*Policy(2,:,:,:,:)+(1-adjust).*(Policy(2,:,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(2,:,:,:,:)<1+n2short+1);
Policyalt(1,:,:,:,:)=Policyalt(1,:,:,:,:)-adjustalt;
Policyalt(2,:,:,:,:)=adjustalt.*Policyalt(2,:,:,:,:)+(1-adjustalt).*(Policyalt(2,:,:,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];

end
