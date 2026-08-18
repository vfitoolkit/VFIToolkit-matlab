function [Vhat,Policy4,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_DC2A_noz_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic + two standard endogenous assets + ExperienceAssete + SemiExo, Divide-and-Conquer (DC2A over a1prime).
% SemiExo graft of ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_e_raw, following ValueFnIter_FHorz_ExpAsseteSemiExo_DC2A_e_raw.
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% semiz is semi-exogenous; there is no Markov z in this variant; e is i.i.d. start-of-period.
% aprimeFn = aprimeFn(d2, a3, e, ...)   (depends on current e; not on z or semiz)
% Policy4 stores (d12, d3, a1prime, a2prime): row 1 is the composite d12=d1+N_d1*(d2-1), a1prime is the divide-conquered standard asset, a2prime the folded standard asset.
%
% Sophisticated QH over the DC argmax axis:
%   Policy4 (and Vhat) come from the  F + beta0*beta*EV  argmax (QH-perceived).
%   Vunderbar is the  F + beta*EV  RHS GATHERED at that same DC argmax (NOT re-maximised).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Vunderbar.
%
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise semiz and e; =1 loop e (semiz parallel); =2 loop semiz outer / inner-loop e.


N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d12,d3,a1prime,a2prime seperately
Policy4=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
% For the return function we just want the full d=(d1,d2,d3) grid (used in the no-EV sections which vectorise over d3)
n_d23=[n_d2,n_d3];
N_d=prod([n_d1,n_d2,n_d3]);
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)

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


% Preallocate (for the EV sections, which loop over d3; hat=QH-perceived argmax, under=beta-RHS gathered at that argmax)
V_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        dind   =rem(maxindex2-1,N_d)+1;
        d12_ind=rem(dind-1,N_d12)+1;
        Vhat(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy4(1,curraindex,:,:,N_j)=d12_ind; % d12 (composite d1,d2)
        Policy4(2,curraindex,:,:,N_j)=ceil(dind/N_d12); % d3
        aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
        Policy4(3,curraindex,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
        Policy4(4,curraindex,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime

        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind      =rem(maxindex-1,N_d)+1;
                a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind + N_d*N_a2*N_a2*N_a3*N_semiz*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4(1,curraindex,:,:,N_j)=d12_ind; % d12 (composite d1,d2)
                Policy4(2,curraindex,:,:,N_j)=ceil(dind/N_d12); % d3
                Policy4(3,curraindex,:,:,N_j)=a1prime_rec; % a1prime
                Policy4(4,curraindex,:,:,N_j)=a2pind; % a2prime
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind   =rem(maxindex-1,N_d)+1;
                a2pind =floor((maxindex-1)/N_d)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind + N_d*N_a2*N_a2*N_a3*N_semiz*eind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4(1,curraindex,:,:,N_j)=d12_ind; % d12 (composite d1,d2)
                Policy4(2,curraindex,:,:,N_j)=ceil(dind/N_d12); % d3
                Policy4(3,curraindex,:,:,N_j)=loweredge(loweredge_idx); % a1prime
                Policy4(4,curraindex,:,:,N_j)=a2pind; % a2prime
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            dind   =rem(maxindex2-1,N_d)+1;
            d12_ind=rem(dind-1,N_d12)+1;
            Vhat(curraindex,:,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy4(1,curraindex,:,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
            Policy4(2,curraindex,:,e_c,N_j)=ceil(dind/N_d12); % d3
            aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
            Policy4(3,curraindex,:,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
            Policy4(4,curraindex,:,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind      =rem(maxindex-1,N_d)+1;
                    a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4(1,curraindex,:,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy4(2,curraindex,:,e_c,N_j)=ceil(dind/N_d12); % d3
                    Policy4(3,curraindex,:,e_c,N_j)=a1prime_rec; % a1prime
                    Policy4(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind   =rem(maxindex-1,N_d)+1;
                    a2pind =floor((maxindex-1)/N_d)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*semizind;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4(1,curraindex,:,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
                    Policy4(2,curraindex,:,e_c,N_j)=ceil(dind/N_d12); % d3
                    Policy4(3,curraindex,:,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                    Policy4(4,curraindex,:,e_c,N_j)=a2pind; % a2prime
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                dind   =rem(maxindex2-1,N_d)+1;
                d12_ind=rem(dind-1,N_d12)+1;
                Vhat(curraindex,z_c,e_c,N_j)       =shiftdim(Vtempii,1);
                Policy4(1,curraindex,z_c,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
                Policy4(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12); % d3
                aprimeind=ceil(maxindex2/N_d); % this is the joint (a1prime,a2prime)
                Policy4(3,curraindex,z_c,e_c,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
                Policy4(4,curraindex,z_c,e_c,N_j)=ceil(aprimeind/N_a1); % a2prime

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind      =rem(maxindex-1,N_d)+1;
                        a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4(1,curraindex,z_c,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
                        Policy4(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12); % d3
                        Policy4(3,curraindex,z_c,e_c,N_j)=a1prime_rec; % a1prime
                        Policy4(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, special_n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind   =rem(maxindex-1,N_d)+1;
                        a2pind =floor((maxindex-1)/N_d)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat;
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4(1,curraindex,z_c,e_c,N_j)=d12_ind; % d12 (composite d1,d2)
                        Policy4(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12); % d3
                        Policy4(3,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx); % a1prime
                        Policy4(4,curraindex,z_c,e_c,N_j)=a2pind; % a2prime
                    end
                end
            end
        end
    end

    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
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
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_hat,[],2);
            entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1,1);
            entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]);
            M=vfoptions.level1n*N_a2*N_a3;
            maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1) + (N_d12*N_a1*N_a2)*M*shiftdim((0:N_semiz-1),-1) + (N_d12*N_a1*N_a2)*M*N_semiz*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
            V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
            Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1) + firstdim*Mblock*N_semiz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind + N_d12*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d12*1*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1) + firstdim*Mblock*N_semiz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind + N_d12*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,:,:,e_c);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat_e,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_hat,[],2);
                entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
                [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under_e,N_d1,1,1,1,1,1,1);
                entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
                M=vfoptions.level1n*N_a2*N_a3;
                maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1) + (N_d12*N_a1*N_a2)*M*shiftdim((0:N_semiz-1),-1);
                Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        firstdim=N_d12*1*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_under_ze=DiscountedEV_under(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under_ze,N_d1,1,1,1,1,1,1);
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                    M=vfoptions.level1n*N_a2*N_a3;
                    maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                            Mblock=level1iidiff(ii)*N_a2*N_a3;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_ze(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_ze(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind      =rem(maxindex-1,N_d12)+1;
                            a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            firstdim=N_d12*1*N_a2;
                            Mblock=level1iidiff(ii)*N_a2*N_a3;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_ze(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_ze(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind   =rem(maxindex-1,N_d12)+1;
                            a2pind =floor((maxindex-1)/N_d12)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4); % max over d3
    Vhat(:,:,:,N_j)=Vhat_jj;
    Policy4(2,:,:,:,N_j)=shiftdim(d3maxindex,-1); % d3 is just maxindex
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy4(1,:,:,:,N_j)=rem(d12aprime_ind-1,N_d12)+1; % d12 (composite d1,d2)
    aprimeind=ceil(d12aprime_ind/N_d12); % this is the joint (a1prime,a2prime)
    Policy4(3,:,:,:,N_j)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4(4,:,:,:,N_j)=ceil(aprimeind/N_a1); % a2prime
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[N_a,N_semiz,N_e]);
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

    % Continuation value is Vunderbar, integrated over e
    EVpre=squeeze(sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_semiz]

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
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_hat,[],2);
            entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1,1);
            entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz,N_e]);
            M=vfoptions.level1n*N_a2*N_a3;
            maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1) + (N_d12*N_a1*N_a2)*M*shiftdim((0:N_semiz-1),-1) + (N_d12*N_a1*N_a2)*M*N_semiz*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
            V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
            Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1) + firstdim*Mblock*N_semiz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind      =rem(maxindex-1,N_d12)+1;
                    a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                    a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind + N_d12*N_a2*N_a2*N_a3*N_semiz*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5) + N_d2*N_a1*N_a2*N_a3*N_semiz*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d12*1*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_semiz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1) + firstdim*Mblock*N_semiz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind   =rem(maxindex-1,N_d12)+1;
                    a2pind =floor((maxindex-1)/N_d12)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind + N_d12*N_a2*N_a2*N_a3*N_semiz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,:,:,e_c);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat_e,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_hat,[],2);
                entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
                [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under_e,N_d1,1,1,1,1,1,1);
                entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_semiz]);
                M=vfoptions.level1n*N_a2*N_a3;
                maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1) + (N_d12*N_a1*N_a2)*M*shiftdim((0:N_semiz-1),-1);
                Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind      =rem(maxindex-1,N_d12)+1;
                        a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                        a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        firstdim=N_d12*1*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_semiz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind   =rem(maxindex-1,N_d12)+1;
                        a2pind =floor((maxindex-1)/N_d12)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat + N_d12*N_a2*N_a2*N_a3*semizind;
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                    end
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
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
            entireEV=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz,N_e]);
            DiscountedEV_under=beta*entireEV;    % exponential
            DiscountedEV_hat=beta0beta*entireEV; % QH-perceived

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_under_ze=DiscountedEV_under(:,:,:,:,:,:,z_c,e_c);
                    DiscountedEV_hat_ze=DiscountedEV_hat(:,:,:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_hat=ReturnMatrix_ii_d3+repelem(DiscountedEV_hat_ze,N_d1,1,1,1,1,1,1);
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3+repelem(DiscountedEV_under_ze,N_d1,1,1,1,1,1,1);
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d12*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                    M=vfoptions.level1n*N_a2*N_a3;
                    maxindexfull=maxindex2 + (N_d12*N_a1*N_a2)*(0:M-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);
                    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                             +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
                    V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                                 +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            firstdim=N_d12*(maxgap(ii)+1)*N_a2;
                            Mblock=level1iidiff(ii)*N_a2*N_a3;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_ze(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_ze(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind      =rem(maxindex-1,N_d12)+1;
                            a1localind=rem(floor((maxindex-1)/N_d12),maxgap(ii)+1)+1;
                            a2pind    =floor((maxindex-1)/(N_d12*(maxgap(ii)+1)))+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d12*(a1prime_rec-1) + N_d12*N_a1*(a2pind-1),1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d123_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_gridvals, z_val, e_val, ReturnFnParamsVec, 3);
                            d2aprime=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                            firstdim=N_d12*1*N_a2;
                            Mblock=level1iidiff(ii)*N_a2*N_a3;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat_ze(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under_ze(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind   =rem(maxindex-1,N_d12)+1;
                            a2pind =floor((maxindex-1)/N_d12)+1;
                            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                            loweredge_idx=dind + N_d12*(a2pind-1) + N_d12*N_a2*a2ind_flat + N_d12*N_a2*N_a2*a3ind_flat;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(dind + N_d12*(loweredge(loweredge_idx)-1) + N_d12*N_a1*(a2pind-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4); % max over d3
    Vhat(:,:,:,jj)=Vhat_jj;
    Policy4(2,:,:,:,jj)=shiftdim(d3maxindex,-1); % d3 is just maxindex
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d3 that corresponds, make it this shape for addition just below
    d12aprime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy4(1,:,:,:,jj)=rem(d12aprime_ind-1,N_d12)+1; % d12 (composite d1,d2)
    aprimeind=ceil(d12aprime_ind/N_d12); % this is the joint (a1prime,a2prime)
    Policy4(3,:,:,:,jj)=rem(aprimeind-1,N_a1)+1; % a1prime
    Policy4(4,:,:,:,jj)=ceil(aprimeind/N_a1); % a2prime
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[N_a,N_semiz,N_e]);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron


end
