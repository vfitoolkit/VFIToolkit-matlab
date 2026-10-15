function [V,Policy3]=ValueFnIter_FHorz_ExpAssetsemiz_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetsemiz_raw (d1, z, noe).
% a = a2 (the experience asset is the only endogenous state)
% semiz is semi-exog state (drives the asset), z is an ordinary Markov; bothz=[semiz,z] with semiz the fast index
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% Policy3 stores (d1, d2, d3) -- no a1prime channel since noa1.

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d12=N_d1*N_d2;
N_a2=prod(n_a2);
N_a=N_a2;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % [N_a2, l_a2]; a2_grid stays STACKED for the
% aprimeFn builders, which split it themselves, but the ReturnFn matrix builders want gridvals
N_z=prod(n_z);
N_semiz=prod(n_semiz);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_semizblock=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offsets for linear indexing
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);
semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, n_bothz, d123_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
        Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1);
        Policy3(3,:,:,N_j)=ceil(maxindex/N_d12);
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, special_n_semizblock, d123_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,semizblock,N_j)=shiftdim(Vtemp,1);
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,semizblock,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,semizblock,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,semizblock,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, special_n_bothz, d123_gridvals, a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,z_c,N_j)=ceil(maxindex/N_d12);
        end
    end
else
    % aprime depends on (d2, a2, current_semiz); independent of d3 and of z -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]

    % noa1: aprimeIndex/Plus1Index reduce to a2primeIndex(+1) directly (no N_a1*... combination)
    if length(n_a2)==1
        aprimeIndex=a2primeIndex;        % [N_d2, N_a2, N_semiz]
        aprimeplus1Index=a2primeIndex+1;
        aprimeProbs_d2a2semiz=a2primeProbs;
        aprimeIndex_full=repmat(aprimeIndex,1,1,N_z); % [N_d2, N_a2, N_bothz] (semiz fast, so tile over z)
        aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
        aprimeProbs_full=repmat(aprimeProbs_d2a2semiz,1,1,N_z);
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
        aIdxK_ll=a2k_ll;
        aIdxK_hl=a2k_hl;
        aIdxK_lh=a2k_lh;
        aIdxK_hh=a2k_hh;
        prK_1=p_1;
        prK_2=p_2;
        aIdx_ll=repmat(aIdxK_ll,1,1,N_z);
        aIdx_hl=repmat(aIdxK_hl,1,1,N_z);
        aIdx_lh=repmat(aIdxK_lh,1,1,N_z);
        aIdx_hh=repmat(aIdxK_hh,1,1,N_z);
        pr1_full=repmat(prK_1,1,1,N_z);
        pr2_full=repmat(prK_2,1,1,N_z);
    end

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
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

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semizblock, d123_gridvals_val, a2_gridvals, z_valblock, ReturnFnParamsVec);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex+semizblock_offset);       % aprimeIndex is [N_d2,N_a2,N_semiz]
                    EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_d2a2semiz;
                    aprimeProbs_z(skipinterp)=0;
                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=EV_2D(aIdxK_ll+semizblock_offset);
                    k2_Vhl=EV_2D(aIdxK_hl+semizblock_offset);
                    k2_Vlh=EV_2D(aIdxK_lh+semizblock_offset);
                    k2_Vhh=EV_2D(aIdxK_hh+semizblock_offset);
                    k2_p1=prK_1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=prK_1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=prK_2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV_z=k2_clo+k2_chi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1; % aprime depends on semiz (fast index of bothz)
                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,semiz_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                    aprimeProbs_slice=aprimeProbs_d2a2semiz(:,:,semiz_part);
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
                    EV1=reshape(EV_z(aprime_slice),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2,N_a2]);

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
                    k2_Vll=reshape(EV_z(k2_sll),[N_d2,N_a2]);
                    k2_Vhl=reshape(EV_z(k2_shl),[N_d2,N_a2]);
                    k2_Vlh=reshape(EV_z(k2_slh),[N_d2,N_a2]);
                    k2_Vhh=reshape(EV_z(k2_shh),[N_d2,N_a2]);
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

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 and unpack policy
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2 (no a1prime split)
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
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]

    if length(n_a2)==1
        aprimeIndex=a2primeIndex;
        aprimeplus1Index=a2primeIndex+1;
        aprimeProbs_d2a2semiz=a2primeProbs;
        aprimeIndex_full=repmat(aprimeIndex,1,1,N_z);
        aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
        aprimeProbs_full=repmat(aprimeProbs_d2a2semiz,1,1,N_z);
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
        aIdxK_ll=a2k_ll;
        aIdxK_hl=a2k_hl;
        aIdxK_lh=a2k_lh;
        aIdxK_hh=a2k_hh;
        prK_1=p_1;
        prK_2=p_2;
        aIdx_ll=repmat(aIdxK_ll,1,1,N_z);
        aIdx_hl=repmat(aIdxK_hl,1,1,N_z);
        aIdx_lh=repmat(aIdxK_lh,1,1,N_z);
        aIdx_hh=repmat(aIdxK_hh,1,1,N_z);
        pr1_full=repmat(prK_1,1,1,N_z);
        pr2_full=repmat(prK_2,1,1,N_z);
    end

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

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

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semizblock, d123_gridvals_val, a2_gridvals, z_valblock, ReturnFnParamsVec);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex+semizblock_offset);
                    EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_d2a2semiz;
                    aprimeProbs_z(skipinterp)=0;
                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: four a2 corners, nested two-corner interp (dim 1 inner, dim 2 outer) with
                    % skipinterp at each level and per-contribution NaN cleanup, so a zero weight against
                    % an infinite node (0*(-Inf)=NaN) cannot poison the sum.
                    k2_Vll=EV_2D(aIdxK_ll+semizblock_offset);
                    k2_Vhl=EV_2D(aIdxK_hl+semizblock_offset);
                    k2_Vlh=EV_2D(aIdxK_lh+semizblock_offset);
                    k2_Vhh=EV_2D(aIdxK_hh+semizblock_offset);
                    k2_p1=prK_1; k2_p1(k2_Vll==k2_Vhl)=0;
                    k2_cll=k2_p1.*k2_Vll; k2_cll(isnan(k2_cll))=0;
                    k2_chl=(1-k2_p1).*k2_Vhl; k2_chl(isnan(k2_chl))=0;
                    k2_lo=k2_cll+k2_chl;
                    k2_p1b=prK_1; k2_p1b(k2_Vlh==k2_Vhh)=0;
                    k2_clh=k2_p1b.*k2_Vlh; k2_clh(isnan(k2_clh))=0;
                    k2_chh=(1-k2_p1b).*k2_Vhh; k2_chh(isnan(k2_chh))=0;
                    k2_hi=k2_clh+k2_chh;
                    k2_p2=prK_2; k2_p2(k2_lo==k2_hi)=0;
                    k2_clo=k2_p2.*k2_lo; k2_clo(isnan(k2_clo))=0;
                    k2_chi=(1-k2_p2).*k2_hi; k2_chi(isnan(k2_chi))=0;
                    entireEV_z=k2_clo+k2_chi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,semiz_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                    aprimeProbs_slice=aprimeProbs_d2a2semiz(:,:,semiz_part);
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
                    EV1=reshape(EV_z(aprime_slice),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2,N_a2]);

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
                    k2_Vll=reshape(EV_z(k2_sll),[N_d2,N_a2]);
                    k2_Vhl=reshape(EV_z(k2_shl),[N_d2,N_a2]);
                    k2_Vlh=reshape(EV_z(k2_slh),[N_d2,N_a2]);
                    k2_Vhh=reshape(EV_z(k2_shh),[N_d2,N_a2]);
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

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 and unpack policy
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1); % d2

end


end
