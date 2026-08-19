function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_DC2A_nod1_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% nod1 (no d1 decision) analog of ValueFnIter_FHorz_ExpAssetzSemiExo_DC2A_e_raw (two standard endogenous assets + experienceassetz + semi-exog state).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% aprimeFn = aprimeFn(d2, a3, z, ...)   (depends on current z, but not on semiz or e)
% Naive quasi-hyperbolic; no a1prime channel since noa1.
% Vtilde/Policy come from the F+beta0*beta*EV argmax; Valt/Policyalt from the F+beta*EV argmax
% (the exponential value, which drives the backward recursion).
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policyalt=zeros(4,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
% For the return function we just want the full d=(d2,d3) grid (used in the no-EV sections which vectorise over d3)
n_d23=[n_d2,n_d3];
N_d=prod([n_d2,n_d3]);
d_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=(1:1:N_d2)'; % [N_d2,1]

if vfoptions.lowmemory==0
    bothzind=shiftdim((0:1:N_bothz-1),-1);
    eind=shiftdim((0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    bothzind=shiftdim((0:1:N_bothz-1),-1);
elseif vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
    special_n_e=ones(1,length(n_e));
    semizind=shiftdim((0:1:N_semiz-1),-1);
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';

a3_gridvals=CreateGridvals(n_a3,a3_grid,1); % per-dim factored a3 grid for the ReturnFn builder (l_a3==1: same as a3_grid)

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]); % offset that picks out the current-bothz column when indexing EV_2D

% Preallocate (for the EV sections, which loop over d3)
V_ford3_alt=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind   =rem(maxindex2-1,N_d)+1;
        Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        Policyalt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
        Policyalt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
        aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
        Policyalt(3,curraindex,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
        Policyalt(4,curraindex,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d)+1;
                a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind + N_d*N_a2*N_a2*N_a3*N_bothz*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policyalt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policyalt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policyalt(3,curraindex,:,:,N_j)=a1prime_rec; % a1prime
                Policyalt(4,curraindex,:,:,N_j)=a2pind; % a2prime
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d)+1;
                a2pind =floor((maxindex-1)/N_d)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind + N_d*N_a2*N_a2*N_a3*N_bothz*eind;
                Policyalt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policyalt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policyalt(3,curraindex,:,:,N_j)=loweredge(loweredge_idx); % a1prime
                Policyalt(4,curraindex,:,:,N_j)=a2pind; % a2prime
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind   =rem(maxindex2-1,N_d)+1;
            Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policyalt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
            Policyalt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
            aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
            Policyalt(3,curraindex,:,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
            Policyalt(4,curraindex,:,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policyalt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policyalt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policyalt(3,curraindex,:,e_c,N_j)=a1prime_rec; % a1prime
                    Policyalt(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_bothz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                    Policyalt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policyalt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policyalt(3,curraindex,:,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                    Policyalt(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            zind=(1:1:N_semiz)+N_semiz*(z_c-1);
            z_val=bothz_gridvals_J(zind,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind   =rem(maxindex2-1,N_d)+1;
                Valt(curraindex,zind,e_c,N_j)=shiftdim(Vtempii,1);
                Policyalt(1,curraindex,zind,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                Policyalt(2,curraindex,zind,e_c,N_j)=ceil(dind/N_d2); % d3
                aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
                Policyalt(3,curraindex,zind,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
                Policyalt(4,curraindex,zind,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,zind,e_c,N_j)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d)+1;
                        a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policyalt(1,curraindex,zind,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policyalt(2,curraindex,zind,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policyalt(3,curraindex,zind,e_c,N_j)=a1prime_rec; % a1prime
                        Policyalt(4,curraindex,zind,e_c,N_j)=a2pind; % a2prime
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,zind,e_c,N_j)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d)+1;
                        a2pind =floor((maxindex-1)/N_d)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                        Policyalt(1,curraindex,zind,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policyalt(2,curraindex,zind,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policyalt(3,curraindex,zind,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                        Policyalt(4,curraindex,zind,e_c,N_j)=a2pind; % a2prime
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_bothz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind   =rem(maxindex2-1,N_d)+1;
                Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policyalt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                Policyalt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
                Policyalt(3,curraindex,z_c,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
                Policyalt(4,curraindex,z_c,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_bothz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d)+1;
                        a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policyalt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policyalt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policyalt(3,curraindex,z_c,e_c,N_j)=a1prime_rec; % a1prime
                        Policyalt(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_bothz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d)+1;
                        a2pind =floor((maxindex-1)/N_d)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        Policyalt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policyalt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policyalt(3,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                        Policyalt(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    end
                end
            end
        end
    end

    % Terminal period: no continuation, so the QH-perceived value equals the exponential one
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and prob of lower; aprimeFn sees current z, not semiz or e)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeIndex_full     =repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz] (current bothz; semiz varies fastest)
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1),1,1,N_semiz); % probability of lower corner

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_alt;
            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz,N_e]),[],1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_tilde;
            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz,N_e]),[],1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2_tilde,1);


            maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
                    dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                    a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                    a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
                    dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                    a2pind =floor((maxindex_alt-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end

            % --- tilde narrow band ---
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
                    dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                    a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                    a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
                    dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                    a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_alt;
                [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_tilde;
                [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                        dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                        a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                        a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                        dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                        a2pind =floor((maxindex_alt-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end

            % --- tilde narrow band ---
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                        dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                        a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                        a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                        dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                        a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_z_alt;
                    [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                    [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                    V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_z_tilde;
                    [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                    V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                    maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                            a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                            a2pind =floor((maxindex_alt-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end

            % --- tilde narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                            a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                            a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_z_alt;
                    [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                    [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_z_tilde;
                    [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                    maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprime_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                            a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprime_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                            a2pind =floor((maxindex_alt-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end

            % --- tilde narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprime_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                            a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprime_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                            a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    % Max over d3 (alt, exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,N_j)=V_jj;
    Policyalt(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(1,:,:,:,N_j)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policyalt(3,:,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policyalt(4,:,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

    % Max over d3 (tilde, QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,N_j)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy(3,:,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy(4,:,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime
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

    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and prob of lower; aprimeFn sees current z, not semiz or e)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeIndex_full     =repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz] (current bothz; semiz varies fastest)
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(repmat(a3primeProbs,N_a1*N_a2,1,1),1,1,N_semiz); % probability of lower corner

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
            entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_alt;
            [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
            [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz,N_e]),[],1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
            entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_tilde;
            [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
            [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz,N_e]),[],1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2_tilde,1);


            maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_alt(ii)>0
                    loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                    a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
                    dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                    a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                    a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii_alt,1);
                    dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                    a2pind =floor((maxindex_alt-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end

            % --- tilde narrow band ---
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_tilde(ii)>0
                    loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                    a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
                    dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                    a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                    a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz,N_e]);
                    [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii_tilde,1);
                    dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                    a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind + N_d2*N_a2*N_a2*N_a3*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_alt;
                [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_tilde;
                [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap_alt(ii)>0
                        loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                        a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                        ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                        dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                        a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                        a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                        dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                        a2pind =floor((maxindex_alt-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end

            % --- tilde narrow band ---
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap_tilde(ii)>0
                        loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                        a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                        ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                        dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                        a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                        a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                        entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                        [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                        dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                        a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*bothzind;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_z_alt;
                    [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                    [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                    V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_z_tilde;
                    [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                    V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                    maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprimez_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                            a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprimez_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                            a2pind =floor((maxindex_alt-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            Policy_ford3_alt(curraindex,zind,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end

            % --- tilde narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprimez_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                            a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprimez_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprimez_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                            a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                            Policy_ford3_tilde(curraindex,zind,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a,N_bothzprime,N_bothz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_bothz) gives NaN, we want zeros
            EV=sum(EV,2); % sum over bothzprime (semiz transition depends on d3)
            EV_2D=reshape(EV,[N_a,N_bothz]); % columns are current-period bothz
            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2*N_a1*N_a2,N_a3,N_bothz], the lower grid point
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset); % the upper grid point
            skipinterp=(EV1==EV2); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_alt=beta*EVbase_qh;
            DiscountedEV_tilde=beta0beta*EVbase_qh;


            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);

            % --- alt pass (exponential beta) ---
                    entireRHS_ii_d3_alt=ReturnMatrix_ii_d3+DiscountedEV_z_alt;
                    [~,maxindex1_alt]=max(entireRHS_ii_d3_alt,[],2);
                    [Vtempii_alt,maxindex2_alt]=max(reshape(entireRHS_ii_d3_alt,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2_alt,1);


            % --- tilde pass (QH-perceived beta0*beta) ---
                    entireRHS_ii_d3_tilde=ReturnMatrix_ii_d3+DiscountedEV_z_tilde;
                    [~,maxindex1_tilde]=max(entireRHS_ii_d3_tilde,[],2);
                    [Vtempii_tilde,maxindex2_tilde]=max(reshape(entireRHS_ii_d3_tilde,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2_tilde,1);


                    maxgap_alt=squeeze(max(max(max(max(max(max( maxindex1_alt(:,1,:,2:end,:,:,:,:)-maxindex1_alt(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    maxgap_tilde=squeeze(max(max(max(max(max(max( maxindex1_tilde(:,1,:,2:end,:,:,:,:)-maxindex1_tilde(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));

            % --- alt narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_alt(ii)>0
                            loweredge_alt=min(maxindex1_alt(:,1,:,ii,:,:,:,:),N_a1-maxgap_alt(ii));
                            a1primeindexes_alt=loweredge_alt+(0:1:maxgap_alt(ii));
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind_vec + N_d2*(a1primeindexes_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprime_alt),[N_d2*(maxgap_alt(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt      =rem(maxindex_alt-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_alt-1)/N_d2),maxgap_alt(ii)+1)+1;
                            a2pind    =floor((maxindex_alt-1)/(N_d2*(maxgap_alt(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge_alt(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind_alt + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_alt=maxindex1_alt(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_alt=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_alt), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_alt=d2ind_vec + N_d2*(loweredge_alt-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_alt=reshape(ReturnMatrix_ii_d3_alt+DiscountedEV_z_alt(d2aprime_alt),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_alt,maxindex_alt]=max(entireRHS_ii_alt,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_alt,1);
                            dind_alt   =rem(maxindex_alt-1,N_d2)+1;
                            a2pind =floor((maxindex_alt-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_alt + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind_alt + N_d2*(loweredge_alt(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end

            % --- tilde narrow band ---
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap_tilde(ii)>0
                            loweredge_tilde=min(maxindex1_tilde(:,1,:,ii,:,:,:,:),N_a1-maxgap_tilde(ii));
                            a1primeindexes_tilde=loweredge_tilde+(0:1:maxgap_tilde(ii));
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(a1primeindexes_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind_vec + N_d2*(a1primeindexes_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprime_tilde),[N_d2*(maxgap_tilde(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde      =rem(maxindex_tilde-1,N_d2)+1;
                            a1localind=rem(floor((maxindex_tilde-1)/N_d2),maxgap_tilde(ii)+1)+1;
                            a2pind    =floor((maxindex_tilde-1)/(N_d2*(maxgap_tilde(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge_tilde(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge_tilde=maxindex1_tilde(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3_tilde=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d123_gridvals, a1_grid(loweredge_tilde), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime_tilde=d2ind_vec + N_d2*(loweredge_tilde-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii_tilde=reshape(ReturnMatrix_ii_d3_tilde+DiscountedEV_z_tilde(d2aprime_tilde),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii_tilde,maxindex_tilde]=max(entireRHS_ii_tilde,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_tilde,1);
                            dind_tilde   =rem(maxindex_tilde-1,N_d2)+1;
                            a2pind =floor((maxindex_tilde-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind_tilde + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind_tilde + N_d2*(loweredge_tilde(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    % Max over d3 (alt, exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,jj)=V_jj;
    Policyalt(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policyalt(1,:,:,:,jj)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policyalt(3,:,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policyalt(4,:,:,:,jj)=ceil(aprimeind/N_a1); % a2prime

    % Max over d3 (tilde, QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    Vtilde(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy(1,:,:,:,jj)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy(3,:,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy(4,:,:,:,jj)=ceil(aprimeind/N_a1); % a2prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
