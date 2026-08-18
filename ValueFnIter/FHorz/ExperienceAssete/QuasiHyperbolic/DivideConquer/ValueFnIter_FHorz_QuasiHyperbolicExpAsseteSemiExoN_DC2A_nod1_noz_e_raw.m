function [V,Policy4,Valt,Policy4alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoN_DC2A_nod1_noz_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + two standard endogenous assets + experienceassetze + semi-exog state, Divide-and-Conquer (DC2A over a1prime). No d1.
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is semi-exogenous; there is no Markov z in this variant; e is i.i.d. start-of-period.
% aprimeFn = aprimeFn(d2, a3, e, ...)   (depends on current e; not on z or semiz)
% Policy4 (and Policy4alt) store (d2, d3, a1prime, a2prime).
%
% Naive QH dual pass over the DC argmax axis:
%   Valt/Policy4alt maximise  F + beta*EV        (the exponential value)
%   V/Policy4       maximise  F + beta0*beta*EV  (the QH-perceived value)
% Each maximisation is a full divide-and-conquer pass (its own level1/maxgap/level2).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
%
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 loop semiz outer / inner-loop e.


N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Valt=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime,a2prime seperately
Policy4=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');
Policy4alt=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
% For the return function we just want the full d=(d2,d3) grid (used in the no-EV sections which vectorise over d3)
n_d23=[n_d2,n_d3];
N_d=prod([n_d2,n_d3]);
d_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

d2ind_vec=(1:1:N_d2)'; % [N_d2,1]

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

a3_gridvals=CreateGridvals(n_a3,a3_grid,1); % per-dim factored a3 grid for the ReturnFn builder (l_a3==1: same as a3_grid)


% Preallocate per-d3 (alt=exponential, tilde=QH-perceived)
V_ford3_alt=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % Terminal: pure return, single DC pass. No continuation => V=Valt, Policy4=Policy4alt.
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind   =rem(maxindex2-1,N_d)+1;
        Valt(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy4alt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
        Policy4alt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
        aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
        Policy4alt(3,curraindex,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
        Policy4alt(4,curraindex,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d)+1;
                a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind + N_d*N_a2*N_a2*N_a3*N_semiz*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy4alt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy4alt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy4alt(3,curraindex,:,:,N_j)=a1prime_rec; % a1prime
                Policy4alt(4,curraindex,:,:,N_j)=a2pind; % a2prime
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d)+1;
                a2pind =floor((maxindex-1)/N_d)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind + N_d*N_a2*N_a2*N_a3*N_semiz*eind;
                Policy4alt(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy4alt(2,curraindex,:,:,N_j)=ceil(dind/N_d2); % d3
                Policy4alt(3,curraindex,:,:,N_j)=loweredge(loweredge_idx); % a1prime
                Policy4alt(4,curraindex,:,:,N_j)=a2pind; % a2prime
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind   =rem(maxindex2-1,N_d)+1;
            Valt(curraindex,:,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy4alt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
            Policy4alt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
            aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
            Policy4alt(3,curraindex,:,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
            Policy4alt(4,curraindex,:,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy4alt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy4alt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy4alt(3,curraindex,:,e_c,N_j)=a1prime_rec; % a1prime
                    Policy4alt(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    Policy4alt(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                    Policy4alt(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2); % d3
                    Policy4alt(3,curraindex,:,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                    Policy4alt(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind   =rem(maxindex2-1,N_d)+1;
                Valt(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
                Policy4alt(3,curraindex,z_c,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
                Policy4alt(4,curraindex,z_c,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d)+1;
                        a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy4alt(3,curraindex,z_c,e_c,N_j)=a1prime_rec; % a1prime
                        Policy4alt(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d)+1;
                        a2pind =floor((maxindex-1)/N_d)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1; % d2
                        Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2); % d3
                        Policy4alt(3,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                        Policy4alt(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    end
                end
            end
        end
    end
    % Terminal period: no continuation, so QH-perceived value equals exponential value
    V(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy4(:,:,:,:,N_j)=Policy4alt(:,:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (lower grid point index and prob of lower; aprimeFn sees current e, not semiz nor z)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_d2a1a2a3e=repmat(a3primeProbs,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_e], probability of lower grid point

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);       % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]); % QH-perceived

            %% alt pass (exponential: F + beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt_e;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_e(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_e(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde_e;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_e(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_e(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_alt_ze=DiscountedEV_alt(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt_ze;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_ze(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_tilde_ze=DiscountedEV_tilde(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde_ze;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_ze(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential), and keep the policy that corresponded to that
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,N_j)=V_jj;
    Policy4alt(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy4alt(1,:,:,:,N_j)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy4alt(3,:,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4alt(4,:,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    V(:,:,:,N_j)=V_jj;
    Policy4(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy4(1,:,:,:,N_j)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy4(3,:,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4(4,:,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime
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

    % Continuation value is the exponential value (Valt), integrated over e'
    EVpre=squeeze(sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (lower grid point index and prob of lower; aprimeFn sees current e, not semiz nor z)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_d2a1a2a3e=repmat(a3primeProbs,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_e], probability of lower grid point

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde;
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind + N_d2*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt_e;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_e(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_e(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde_e;
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_e(d2aprimez),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_e(d2aprimez),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d2_gridvals,d3_grid(d3_c).*ones(N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            EV=EVpre.*shiftdim(pi_semiz_d3',-1); % [N_a,N_semizprime,N_semiz]
            EV(isnan(EV))=0; % multilication of -Inf (in V) and zero (in pi_semiz_d3) gives NaN, we want zeros
            EV=sum(EV,2); % sum over semizprime (semiz transition depends on d3)
            EV_byzcur=reshape(EV,[N_a,N_semiz]); % columns are current-period semiz
            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the lower grid point
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_e,N_semiz]); % the upper grid point
            skipinterp=(Vlower==Vupper); % Note: will only skip when both points are outside the grid (or e.g. -Inf)
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2a3e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2*a1prime*a2prime,a3,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime*a2prime,a3,semizcur,e_cur)
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_alt_ze=DiscountedEV_alt(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_alt_ze;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt_ze(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_tilde_ze=DiscountedEV_tilde(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_tilde_ze;
                    [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_ze(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind      =rem(maxindex-1,N_d2)+1;
                            a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(a1prime_rec-1) + N_d2*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde_ze(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind   =rem(maxindex-1,N_d2)+1;
                            a2pind =floor((maxindex-1)/N_d2)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d2*(loweredge(loweredge_idx)-1) + N_d2*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential), and keep the policy that corresponded to that
    [V_jj,maxindex]=max(V_ford3_alt,[],4); % max over d3
    Valt(:,:,:,jj)=V_jj;
    Policy4alt(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d2aprime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy4alt(1,:,:,:,jj)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy4alt(3,:,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4alt(4,:,:,:,jj)=ceil(aprimeind/N_a1); % a2prime

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy4(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2aprime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy4(1,:,:,:,jj)=rem(d2aprime_ind-1,N_d2)+1; % d2
    aprimeind=ceil(d2aprime_ind/N_d2); % this is the joint (a1prime,a2prime)
    Policy4(3,:,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4(4,:,:,:,jj)=ceil(aprimeind/N_a1); % a2prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
