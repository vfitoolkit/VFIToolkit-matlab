function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_DC2A_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% SemiExo graft of ValueFnIter_FHorz_ExpAssetz_DC2A_raw (two standard endogenous assets + experienceassetz + semi-exog state, no e shock).
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% aprimeFn = aprimeFn(d2, a3, z, ...)   (depends on current z, but not on semiz)
% Sophisticated quasi-hyperbolic; no a1prime channel since noa1.
% Vhat/Policy come from the F+beta0*beta*EV argmax; Vunderbar is the F+beta*EV RHS
% GATHERED at that same argmax (not re-maximised), and drives the backward recursion.
% lowmemory: 2 shocks {z,semiz} => levels {0,1,2}.
%   =0 vectorise bothz; =1 split: outer-loop z / semiz parallel; =2 joint: loop over bothz.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Vhat=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
% For the return function we just want the full d=(d1,d2,d3) grid (used in the no-EV sections which vectorise over d3)
n_d23=[n_d2,n_d3];
N_d=prod([n_d1,n_d2,n_d3]);
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)

if vfoptions.lowmemory==0
    bothzind=shiftdim((0:1:N_bothz-1),-1);
elseif vfoptions.lowmemory==1
    special_n_semiz=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
    semizind=shiftdim((0:1:N_semiz-1),-1);
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]); % offset that picks out the current-bothz column when indexing EV_2D

% Preallocate (for the EV sections, which loop over d3)
V_ford3_hat=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind   =rem(maxindex2-1,N_d)+1;
        d12_ind=rem(dind-1,N_d12)+1;
        Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,curraindex,:,N_j)=d12_ind; % d12 (composite d1,d2)
        Policy(2,curraindex,:,N_j)=ceil(dind/N_d12); % d3
        aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
        Policy(3,curraindex,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
        Policy(4,curraindex,:,N_j)=ceil(aprimeind/N_a1); % a2prime

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d)+1;
                a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy(1,curraindex,:,N_j)=d12_ind; % d12 (composite d1,d2)
                Policy(2,curraindex,:,N_j)=ceil(dind/N_d12); % d3
                Policy(3,curraindex,:,N_j)=a1prime_rec; % a1prime
                Policy(4,curraindex,:,N_j)=a2pind; % a2prime
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d)+1;
                a2pind =floor((maxindex-1)/N_d)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy(1,curraindex,:,N_j)=d12_ind; % d12 (composite d1,d2)
                Policy(2,curraindex,:,N_j)=ceil(dind/N_d12); % d3
                Policy(3,curraindex,:,N_j)=loweredge(loweredge_idx); % a1prime
                Policy(4,curraindex,:,N_j)=a2pind; % a2prime
            end
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for z_c=1:N_z
            zind=(1:1:N_semiz)+N_semiz*(z_c-1);
            z_val=bothz_gridvals_J(zind,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind   =rem(maxindex2-1,N_d)+1;
            d12_ind=rem(dind-1,N_d12)+1;
            Vhat(curraindex,zind,N_j)=shiftdim(Vtempii,1);
            Policy(1,curraindex,zind,N_j)=d12_ind; % d12 (composite d1,d2)
            Policy(2,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
            aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
            Policy(3,curraindex,zind,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
            Policy(4,curraindex,zind,N_j)=ceil(aprimeind/N_a1); % a2prime

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,zind,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy(1,curraindex,zind,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy(2,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
                    Policy(3,curraindex,zind,N_j)=a1prime_rec; % a1prime
                    Policy(4,curraindex,zind,N_j)=a2pind; % a2prime
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,zind,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy(1,curraindex,zind,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy(2,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
                    Policy(3,curraindex,zind,N_j)=loweredge(loweredge_idx); % a1prime
                    Policy(4,curraindex,zind,N_j)=a2pind; % a2prime
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind   =rem(maxindex2-1,N_d)+1;
            d12_ind=rem(dind-1,N_d12)+1;
            Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,N_j)=d12_ind; % d12 (composite d1,d2)
            Policy(2,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
            aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
            Policy(3,curraindex,z_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
            Policy(4,curraindex,z_c,N_j)=ceil(aprimeind/N_a1); % a2prime

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy(1,curraindex,z_c,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy(2,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
                    Policy(3,curraindex,z_c,N_j)=a1prime_rec; % a1prime
                    Policy(4,curraindex,z_c,N_j)=a2pind; % a2prime
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy(1,curraindex,z_c,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy(2,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
                    Policy(3,curraindex,z_c,N_j)=loweredge(loweredge_idx); % a1prime
                    Policy(4,curraindex,z_c,N_j)=a2pind; % a2prime
                end
            end
        end
    end

    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and prob of lower; aprimeFn sees current z, not semiz)

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
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1)+shiftdim((N_d12*N_a1*N_a2)*(vfoptions.level1n*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,:,d3_c)=shiftdim(Vtempii_under,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*(maxgap(ii)+1)*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
                    V_ford3_under(curraindex,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*1*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
                    V_ford3_under(curraindex,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,zind);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,zind);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_hat,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_z_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1)+shiftdim((N_d12*N_a1*N_a2)*(vfoptions.level1n*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,zind,d3_c)=shiftdim(Vtempii_under,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*(maxgap(ii)+1)*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
                        V_ford3_under(curraindex,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*1*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
                        V_ford3_under(curraindex,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,z_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_hat,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_z_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,z_c,d3_c)=shiftdim(Vtempii_under,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1);
                        V_ford3_under(curraindex,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1);
                        V_ford3_under(curraindex,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,N_j)=rem(d12aprime_ind-1,N_d12)+1; % d12 (composite d1,d2)
    aprimeind=ceil(d12aprime_ind/N_d12); % this is the joint (a1prime,a2prime)
    Policy(3,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy(4,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
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

    EVpre=Vunderbar(:,:,jj+1); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower-corner index and prob of lower; aprimeFn sees current z, not semiz)

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
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1)+shiftdim((N_d12*N_a1*N_a2)*(vfoptions.level1n*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,:,d3_c)=shiftdim(Vtempii_under,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*(maxgap(ii)+1)*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
                    V_ford3_under(curraindex,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*1*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_bothz)-1),-1);
                    V_ford3_under(curraindex,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_hat(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,zind);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,zind);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_hat,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_z_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1)+shiftdim((N_d12*N_a1*N_a2)*(vfoptions.level1n*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,zind,d3_c)=shiftdim(Vtempii_under,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*(maxgap(ii)+1)*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
                        V_ford3_under(curraindex,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1)+shiftdim((N_d12*1*N_a2)*(level1iidiff(ii)*N_a2*N_a3)*(0:1:(N_semiz)-1),-1);
                        V_ford3_under(curraindex,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_hat(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat  =beta0beta*EVbase_qh;


            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,z_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_hat,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            % Vunderbar: value of the hat-policy under the long-run beta (gather, no second argmax)
            entireRHS_under_flat=reshape(ReturnMatrix_ii_d3+repelem(DiscountedEV_z_under,N_d1,1,1,1,1,1,1),[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
            maxindexfull=maxindex2+(N_d12*N_a1*N_a2)*(0:1:(vfoptions.level1n*N_a2*N_a3)-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);
            V_ford3_under(curraindex,z_c,d3_c)=shiftdim(Vtempii_under,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        maxindexfull=maxindex+(N_d12*(maxgap(ii)+1)*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1);
                        V_ford3_under(curraindex,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_hat(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_under(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        maxindexfull=maxindex+(N_d12*1*N_a2)*(0:1:(level1iidiff(ii)*N_a2*N_a3)-1);
                        V_ford3_under(curraindex,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                        V_ford3_hat(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_hat(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,jj)=rem(d12aprime_ind-1,N_d12)+1; % d12 (composite d1,d2)
    aprimeind=ceil(d12aprime_ind/N_d12); % this is the joint (a1prime,a2prime)
    Policy(3,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy(4,:,:,jj)=ceil(aprimeind/N_a1); % a2prime

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
