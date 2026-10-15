function [V,Policy3]=ValueFnIter_FHorz_ExpAssetsemiz_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate

n_bothz=n_semiz;

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_bothz=N_semiz;
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
% Policy storage with separate entries for d2, d3, a1prime (no d1)
Policy3=zeros(3,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=semiz_gridvals_J;

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy3(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy3(2,:,:,:,N_j)=ceil(d_ind/N_d2); % d3
        Policy3(3,:,:,:,N_j)=ceil(maxindex/N_d23); % a1prime
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy3(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
            Policy3(3,:,:,e_c,N_j)=ceil(maxindex/N_d23);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy3(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy3(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
                Policy3(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d23);
            end
        end
    end
else
    % aprime depends on (d2, a1, a2, current_z); independent of d3, semiz, e -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
        aprimeIndex_full=aprimeIndex;
        aprimeplus1Index_full=aprimeplus1Index;
        aprimeProbs_full=aprimeProbs_d2a1a2semiz;
    else
        % l_a2==2: the builder returns [l_a2,N_d2,N_a2,N_semiz] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, then push each corner
        % through THIS FILE'S OWN construction so the a1-Kron and any tiling over z are applied
        % exactly as the l_a2==1 arm applies them.
        n_a2_1=n_a2(1);
        lo_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_semiz]);
        lo_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_semiz]);
        p_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_semiz]);
        p_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_semiz]);
        a2k_ll=lo_1+n_a2_1*(lo_2-1);
        a2k_hl=(lo_1+1)+n_a2_1*(lo_2-1);
        a2k_lh=lo_1+n_a2_1*lo_2;
        a2k_hh=(lo_1+1)+n_a2_1*lo_2;
        aIdxK_ll=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_ll-1,N_a1,1,1);
        aIdxK_hl=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hl-1,N_a1,1,1);
        aIdxK_lh=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_lh-1,N_a1,1,1);
        aIdxK_hh=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hh-1,N_a1,1,1);
        prK_1=repmat(p_1,N_a1,1,1);
        prK_2=repmat(p_2,N_a1,1,1);
        aIdx_ll=aIdxK_ll;
        aIdx_hl=aIdxK_hl;
        aIdx_lh=aIdxK_lh;
        aIdx_hh=aIdxK_hh;
        pr1_full=prK_1;
        pr2_full=prK_2;
    end

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1); % broadcasts over e

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);

                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV;

                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,semiz_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                    aprimeProbs_slice=aprimeProbs_d2a1a2semiz(:,:,semiz_part);
                else
                    % slice the four corner arrays and the two per-dim probs with THIS FILE'S OWN subscript
                    k2_sll=aIdxK_ll(:,:,semiz_part);
                    k2_shl=aIdxK_hl(:,:,semiz_part);
                    k2_slh=aIdxK_lh(:,:,semiz_part);
                    k2_shh=aIdxK_hh(:,:,semiz_part);
                    k2_prs1=prK_1(:,:,semiz_part);
                    k2_prs2=prK_2(:,:,semiz_part);
                end

                if length(n_a2)==1
                    EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_slice;
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=reshape(EV_z(k2_sll),[N_d2*N_a1,N_a2]);
                    k2_Vhl=reshape(EV_z(k2_shl),[N_d2*N_a1,N_a2]);
                    k2_Vlh=reshape(EV_z(k2_slh),[N_d2*N_a1,N_a2]);
                    k2_Vhh=reshape(EV_z(k2_shh),[N_d2*N_a1,N_a2]);
                    k2_p1=k2_prs1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=k2_prs1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=k2_prs2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV_z=k2_clo+k2_chi;
                end
                DiscountedEV_z=DiscountFactorParamsVec*repelem(entireEV_z,1,N_a1);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z;

                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    % Max over d3 (dim 4)
    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
        aprimeIndex_full=aprimeIndex;
        aprimeplus1Index_full=aprimeplus1Index;
        aprimeProbs_full=aprimeProbs_d2a1a2semiz;
    else
        % l_a2==2: the builder returns [l_a2,N_d2,N_a2,N_semiz] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, then push each corner
        % through THIS FILE'S OWN construction so the a1-Kron and any tiling over z are applied
        % exactly as the l_a2==1 arm applies them.
        n_a2_1=n_a2(1);
        lo_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_semiz]);
        lo_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_semiz]);
        p_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_semiz]);
        p_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_semiz]);
        a2k_ll=lo_1+n_a2_1*(lo_2-1);
        a2k_hl=(lo_1+1)+n_a2_1*(lo_2-1);
        a2k_lh=lo_1+n_a2_1*lo_2;
        a2k_hh=(lo_1+1)+n_a2_1*lo_2;
        aIdxK_ll=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_ll-1,N_a1,1,1);
        aIdxK_hl=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hl-1,N_a1,1,1);
        aIdxK_lh=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_lh-1,N_a1,1,1);
        aIdxK_hh=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2k_hh-1,N_a1,1,1);
        prK_1=repmat(p_1,N_a1,1,1);
        prK_2=repmat(p_2,N_a1,1,1);
        aIdx_ll=aIdxK_ll;
        aIdx_hl=aIdxK_hl;
        aIdx_lh=aIdxK_lh;
        aIdx_hh=aIdxK_hh;
        pr1_full=prK_1;
        pr2_full=prK_2;
    end

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                k2_Vll=EV_2D(aIdx_ll+bothz_offset);
                k2_Vhl=EV_2D(aIdx_hl+bothz_offset);
                k2_Vlh=EV_2D(aIdx_lh+bothz_offset);
                k2_Vhh=EV_2D(aIdx_hh+bothz_offset);
                k2_p1=pr1_full; k2_p1(k2_Vll==k2_Vhl)=0;
                k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                k2_lo=k2_cll+k2_chl;
                k2_p1b=pr1_full; k2_p1b(k2_Vlh==k2_Vhh)=0;
                k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                k2_hi=k2_clh+k2_chh;
                k2_p2=pr2_full; k2_p2(k2_lo==k2_hi)=0;
                k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                entireEV=k2_clo+k2_chi;
            end
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0);

                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV;

                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,semiz_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                    aprimeProbs_slice=aprimeProbs_d2a1a2semiz(:,:,semiz_part);
                else
                    % slice the four corner arrays and the two per-dim probs with THIS FILE'S OWN subscript
                    k2_sll=aIdxK_ll(:,:,semiz_part);
                    k2_shl=aIdxK_hl(:,:,semiz_part);
                    k2_slh=aIdxK_lh(:,:,semiz_part);
                    k2_shh=aIdxK_hh(:,:,semiz_part);
                    k2_prs1=prK_1(:,:,semiz_part);
                    k2_prs2=prK_2(:,:,semiz_part);
                end

                if length(n_a2)==1
                    EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_slice;
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=reshape(EV_z(k2_sll),[N_d2*N_a1,N_a2]);
                    k2_Vhl=reshape(EV_z(k2_shl),[N_d2*N_a1,N_a2]);
                    k2_Vlh=reshape(EV_z(k2_slh),[N_d2*N_a1,N_a2]);
                    k2_Vhh=reshape(EV_z(k2_shh),[N_d2*N_a1,N_a2]);
                    k2_p1=k2_prs1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=k2_prs1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=k2_prs2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV_z=k2_clo+k2_chi;
                end
                DiscountedEV_z=DiscountFactorParamsVec*repelem(entireEV_z,1,N_a1);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z;

                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2);
end


end
