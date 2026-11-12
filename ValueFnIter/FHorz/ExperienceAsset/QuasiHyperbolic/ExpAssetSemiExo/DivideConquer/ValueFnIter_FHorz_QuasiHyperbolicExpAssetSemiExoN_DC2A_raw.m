function [Vtilde,Policy4,Valt,Policy4alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_DC2A_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% SemiExo graft of ValueFnIter_FHorz_ExpAsset_DC2A_raw (no e shock).
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% Policy4 stores (d1, d2, d3, joint(a1prime,a2prime)); the 4th row is a1prime+N_a1*(a2prime-1).
% lowmemory: 2 shocks {z,semiz} => levels {0,1,2}.
%   =0 vectorise bothz; =1 split: outer-loop z / semiz parallel; =2 joint: loop over bothz.
% Naive quasi-hyperbolic discounting version, Divide-and-Conquer over a1prime with a second folded standard asset (DC2A).
%
% Naive QH dual pass:
%   Valt/Policy4alt maximise  F + beta*EV        (the exponential value; drives the backward recursion)
%   Vtilde/Policy4  maximise  F + beta0*beta*EV  (the QH-perceived value that is reported)
% Each maximisation is a full divide-and-conquer pass (its own level1 sweep, maxgap and narrow band),
% and each has its own per-d3 store, so the max over d3 is also done twice.
% beta0 is received as a trailing input.
% Backward EVpre uses Valt (the exponential continuation value).

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

Valt=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime,a2prime) seperately
Policy4=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray');
Policy4alt=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray');

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

% Preallocate (for the EV sections, which loop over d3)
V_ford3_alt=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz*N_z,N_d3,'gpuArray');

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
        Valt(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy4alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy4alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy4alt(3,curraindex,:,N_j)=ceil(dind/N_d12); % d3
        Policy4alt(4,curraindex,:,N_j)=ceil(maxindex2/N_d); % joint(a1prime,a2prime)

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d)+1;
                a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                Policy4alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1); % d2
                Policy4alt(3,curraindex,:,N_j)=ceil(dind/N_d12); % d3
                Policy4alt(4,curraindex,:,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d)+1;
                a2pind =floor((maxindex-1)/N_d)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*bothzind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                Policy4alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1); % d2
                Policy4alt(3,curraindex,:,N_j)=ceil(dind/N_d12); % d3
                Policy4alt(4,curraindex,:,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
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
            Valt(curraindex,zind,N_j)       =shiftdim(Vtempii,1);
            Policy4alt(1,curraindex,zind,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy4alt(2,curraindex,zind,N_j)=ceil(d12_ind/N_d1); % d2
            Policy4alt(3,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
            Policy4alt(4,curraindex,zind,N_j)=ceil(maxindex2/N_d); % joint(a1prime,a2prime)

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,zind,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,zind,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                    Policy4alt(2,curraindex,zind,N_j)=ceil(d12_ind/N_d1); % d2
                    Policy4alt(3,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
                    Policy4alt(4,curraindex,zind,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,zind,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,zind,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                    Policy4alt(2,curraindex,zind,N_j)=ceil(d12_ind/N_d1); % d2
                    Policy4alt(3,curraindex,zind,N_j)=ceil(dind/N_d12); % d3
                    Policy4alt(4,curraindex,zind,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
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
            Valt(curraindex,z_c,N_j)       =shiftdim(Vtempii,1);
            Policy4alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy4alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1); % d2
            Policy4alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
            Policy4alt(4,curraindex,z_c,N_j)=ceil(maxindex2/N_d); % joint(a1prime,a2prime)

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                    Policy4alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1); % d2
                    Policy4alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
                    Policy4alt(4,curraindex,z_c,N_j)=a1prime_rec+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                    Policy4alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1); % d2
                    Policy4alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12); % d3
                    Policy4alt(4,curraindex,z_c,N_j)=loweredge(loweredge_idx)+N_a1*(a2pind-1); % joint(a1prime,a2prime)
                end
            end
        end
    end

    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy4(:,:,:,N_j)=Policy4alt(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz])); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            %% alt pass (exponential: F + beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz]));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                %% alt pass (exponential: F + beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
                %% tilde pass (QH-perceived: F + beta0*beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz]));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                %% alt pass (exponential: F + beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
                %% tilde pass (QH-perceived: F + beta0*beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end
    end

    % Max over d3 for alt (exponential), and keep the policy that corresponded to that
    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,N_j)=V_jj;
    Policy4alt(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    d12_ind=rem(d12aprime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4alt(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy4alt(4,:,:,N_j)=ceil(d12aprime_ind/N_d12); % joint(a1prime,a2prime)
    % Max over d3 for tilde (QH-perceived), and keep the policy that corresponded to that
    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,N_j)=V_jj;
    Policy4(3,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    d12_ind=rem(d12aprime_ind-1,N_d12)+1;
    Policy4(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy4(4,:,:,N_j)=ceil(d12aprime_ind/N_d12); % joint(a1prime,a2prime)
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

    EVpre=Valt(:,:,jj+1); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV_aprime(aprimeProbs==1)=Vlower(aprimeProbs==1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz]));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            %% alt pass (exponential: F + beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_bothz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz]));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,zind);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,zind);
                %% alt pass (exponential: F + beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_alt(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
                %% tilde pass (QH-perceived: F + beta0*beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprimez),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprimez),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,zind,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_tilde(curraindex,zind,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EVinf=(EV_aprime==-Inf);
            EV=EV_aprime;
            EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV=reshape(EV,[N_d2*N_a1*N_a2*N_a3,N_bothz])*pi_bothz';
            EV(reshape(EVinf,[N_d2*N_a1*N_a2*N_a3,N_bothz])*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV=squeeze(reshape(EV,[N_d2*N_a1*N_a2,N_a3,N_bothz]));
            DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);   % QH-perceived

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_alt=DiscountedEV_alt(:,:,:,:,:,:,z_c);
                DiscountedEV_z_tilde=DiscountedEV_tilde(:,:,:,:,:,:,z_c);
                %% alt pass (exponential: F + beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_alt,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_alt(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
                %% tilde pass (QH-perceived: F + beta0*beta*EV)
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z_tilde,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprime),[N_d12*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_z_tilde(d2aprime),[N_d12*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end
    end

    % Max over d3 for alt (exponential), and keep the policy that corresponded to that
    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_alt,[],3); % max over d3
    Valt(:,:,jj)=V_jj;
    Policy4alt(3,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    d12_ind=rem(d12aprime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4alt(2,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy4alt(4,:,:,jj)=ceil(d12aprime_ind/N_d12); % joint(a1prime,a2prime)
    % Max over d3 for tilde (QH-perceived), and keep the policy that corresponded to that
    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3); % max over d3
    Vtilde(:,:,jj)=V_jj;
    Policy4(3,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    d12_ind=rem(d12aprime_ind-1,N_d12)+1;
    Policy4(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4(2,:,:,jj)=ceil(d12_ind/N_d1); % d2
    Policy4(4,:,:,jj)=ceil(d12aprime_ind/N_d12); % joint(a1prime,a2prime)

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
