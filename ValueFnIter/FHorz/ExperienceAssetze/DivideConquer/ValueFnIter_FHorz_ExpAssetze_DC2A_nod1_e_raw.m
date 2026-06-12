function [V,Policy]=ValueFnIter_FHorz_ExpAssetze_DC2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% _e analog of ValueFnIter_FHorz_ExpAssetze_DC2A_nod1_raw (z-dep aprimeFn (Markov z) and i.i.d. e).
% lowmemory=0 full vectorization; lowmemory=1 loop over z (e vectorized); lowmemory=2 nested z+e.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray');

if vfoptions.lowmemory==0
    zind=shiftdim((0:1:N_z-1),-1);
    eind=shiftdim((0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    eind=shiftdim((0:1:N_e-1),-2); % e at dim 4 of [1,L,1,N_e]
else
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Per-dim factored a3 grid for the ReturnFn builder (l_a3==1: 1 column, l_a3==2: 2 columns)
a3_gridvals=CreateGridvals(n_a3,a3_grid,1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,:,N_j)=d2ind;
        Policy(2,curraindex,:,:,N_j)=a1pind;
        Policy(3,curraindex,:,:,N_j)=a2pind;

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,:,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,:,N_j)=d2ind;
            Policy(2,curraindex,z_c,:,N_j)=a1pind;
            Policy(3,curraindex,z_c,:,N_j)=a2pind;

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=a1prime_rec;
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    V(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
                d2ind  =rem(maxindex2-1,N_d2)+1;
                a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
                a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
                V(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                Policy(2,curraindex,z_c,e_c,N_j)=a1pind;
                Policy(3,curraindex,z_c,e_c,N_j)=a2pind;

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                             + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=a1prime_rec;
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx);
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    end
                end
            end
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_exp=repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,:,N_j)=d2ind;
        Policy(2,curraindex,:,:,N_j)=a1pind;
        Policy(3,curraindex,:,:,N_j)=a2pind;

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,:,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,:,N_j)=d2ind;
            Policy(2,curraindex,z_c,:,N_j)=a1pind;
            Policy(3,curraindex,z_c,:,N_j)=a2pind;

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,1,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=a1prime_rec;
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,1,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze;
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
                d2ind  =rem(maxindex2-1,N_d2)+1;
                a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
                a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
                V(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                Policy(2,curraindex,z_c,e_c,N_j)=a1pind;
                Policy(3,curraindex,z_c,e_c,N_j)=a2pind;

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                             + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=a1prime_rec;
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime),[N_d2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx);
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    end
                end
            end
        end
    end
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

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % l_a3==1: a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z,N_e] (legacy lower-corner)
    % l_a3==2: a3primeIndex/a3primeProbs are [l_a3,N_d2,N_a3,N_z,N_e] (per-dim factored)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);

    if length(n_a3)==1
        a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
        aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
        aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
        aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

        Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a3==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a3_1=n_a3(1);
        loIdx_1_repd=repmat(reshape(a3primeIndex(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        loIdx_2_repd=repmat(reshape(a3primeIndex(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1);
        prob_1_exp=repmat(reshape(a3primeProbs(1,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a3primeProbs(2,:,:,:,:),[N_d2,N_a3,N_z,N_e]),N_a1*N_a2,1,1,1,N_z);

        a3_kron_ll= loIdx_1_repd   +n_a3_1*(loIdx_2_repd-1);
        a3_kron_hl=(loIdx_1_repd+1)+n_a3_1*(loIdx_2_repd-1);
        a3_kron_lh= loIdx_1_repd   +n_a3_1* loIdx_2_repd;
        a3_kron_hh=(loIdx_1_repd+1)+n_a3_1* loIdx_2_repd;

        aprime_ll=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_ll-1);
        aprime_hl=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hl-1);
        aprime_lh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_lh-1);
        aprime_hh=a1_col + N_a1*a2_col + N_a1*N_a2*(a3_kron_hh-1);

        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        V(curraindex,:,:,jj)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,:,jj)=d2ind;
        Policy(2,curraindex,:,:,jj)=a1pind;
        Policy(3,curraindex,:,:,jj)=a2pind;

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,jj)=d2ind;
                Policy(2,curraindex,:,:,jj)=a1prime_rec;
                Policy(3,curraindex,:,:,jj)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,jj)=d2ind;
                Policy(2,curraindex,:,:,jj)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,jj)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;
            [~,maxindex1]=max(entireRHS_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            V(curraindex,z_c,:,jj)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,:,jj)=d2ind;
            Policy(2,curraindex,z_c,:,jj)=a1pind;
            Policy(3,curraindex,z_c,:,jj)=a2pind;

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,1,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,:,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,:,jj)=d2ind;
                    Policy(2,curraindex,z_c,:,jj)=a1prime_rec;
                    Policy(3,curraindex,z_c,:,jj)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_z(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,1,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,z_c,:,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    Policy(1,curraindex,z_c,:,jj)=d2ind;
                    Policy(2,curraindex,z_c,:,jj)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,:,jj)=a2pind;
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c,:);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze;
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
                d2ind  =rem(maxindex2-1,N_d2)+1;
                a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
                a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
                V(curraindex,z_c,e_c,jj)       =shiftdim(Vtempii,1);
                Policy(1,curraindex,z_c,e_c,jj)=d2ind;
                Policy(2,curraindex,z_c,e_c,jj)=a1pind;
                Policy(3,curraindex,z_c,e_c,jj)=a2pind;

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                             + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy(1,curraindex,z_c,e_c,jj)=d2ind;
                        Policy(2,curraindex,z_c,e_c,jj)=a1prime_rec;
                        Policy(3,curraindex,z_c,e_c,jj)=a2pind;
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime),[N_d2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        Policy(1,curraindex,z_c,e_c,jj)=d2ind;
                        Policy(2,curraindex,z_c,e_c,jj)=loweredge(loweredge_idx);
                        Policy(3,curraindex,z_c,e_c,jj)=a2pind;
                    end
                end
            end
        end
    end
end


end
