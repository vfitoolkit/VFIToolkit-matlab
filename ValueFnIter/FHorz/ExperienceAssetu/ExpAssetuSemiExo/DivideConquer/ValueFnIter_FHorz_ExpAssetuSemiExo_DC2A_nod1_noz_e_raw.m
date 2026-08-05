function [V,Policy3]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC2A_nod1_noz_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, n_u, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, u_gridvals, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% nod1, noz, e+semiz graft of ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_e_raw (which has d1 and Markov z).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz). No d1, no Markov z.
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is the semi-exogenous state (plays the role of bothz as there is no z); e is i.i.d.
% Policy3 stores (d2, d3, joint(a1prime,a2prime)); the 3rd row is a1prime+N_a1*(a2prime-1).
% lowmemory: 2 shocks {semiz,e} => levels {0,1,2}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 outer-loop semiz / inner-loop e.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,joint(a1prime,a2prime) seperately
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
% For the return function we just want the full d=(d2,d3) grid (used in the no-EV section which vectorises over d3)
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

d2ind_vec=(1:1:N_d2)'; % [N_d2,1]; nod1, so d (for the experience asset) is just d2

if vfoptions.lowmemory==0
    semizind=shiftdim((0:1:N_semiz-1),-1);
    eind=shiftdim((0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    semizind=shiftdim((0:1:N_semiz-1),-1);
elseif vfoptions.lowmemory==2
    special_n_semiz=ones(1,length(n_semiz));
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';

% Preallocate (for the EV sections, which loop over d3)
V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d23_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind=rem(maxindex2-1,N_d23)+1;
        V(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
        Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
        Policy3(3,curraindex,:,:,N_j)=ceil(maxindex2/N_d23); % joint(a1prime,a2prime)

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d23_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d23)+1;
                a1localind=rem(floor((maxindex-1)/N_d23),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d23*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat + N_d23*N_a2*N_a2*N_a3*semizind + N_d23*N_a2*N_a2*N_a3*N_semiz*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,:,:,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d23_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d23)+1;
                a2pind =floor((maxindex-1)/N_d23)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat + N_d23*N_a2*N_a2*N_a3*semizind + N_d23*N_a2*N_a2*N_a3*N_semiz*eind;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,:,:,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d23_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d23*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind=rem(maxindex2-1,N_d23)+1;
            V(curraindex,:,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
            Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
            Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex2/N_d23); % joint(a1prime,a2prime)

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d23_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d23)+1;
                    a1localind=rem(floor((maxindex-1)/N_d23),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d23*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat + N_d23*N_a2*N_a2*N_a3*semizind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy3(3,curraindex,:,e_c,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d23_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d23)+1;
                    a2pind =floor((maxindex-1)/N_d23)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat + N_d23*N_a2*N_a2*N_a3*semizind;
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy3(3,curraindex,:,e_c,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d23*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind=rem(maxindex2-1,N_d23)+1;
                V(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex2/N_d23); % joint(a1prime,a2prime)

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d23)+1;
                        a1localind=rem(floor((maxindex-1)/N_d23),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d23*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy3(3,curraindex,z_c,e_c,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d23)+1;
                        a2pind =floor((maxindex-1)/N_d23)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d23*(a2pind-1) + N_d23*N_a2*a2ind_flat + N_d23*N_a2*N_a2*a3ind_flat;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy3(3,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                    end
                end
            end
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2aprime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,N_j)=ceil(d2aprime_ind/N_d2); % joint(a1prime,a2prime)
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=squeeze(sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_semiz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_semiz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz], indexed by semizprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EV=EV_aprime.*shiftdim(pi_semiz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2aprime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,jj)=ceil(d2aprime_ind/N_d2); % joint(a1prime,a2prime)
end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
