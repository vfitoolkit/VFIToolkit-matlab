function [V,Policy4,Valt,Policy4alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoN_DC1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + ExperienceAssetze + SemiExo, Divide-and-Conquer (DC1 over a1prime).
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, z, e, ...)   (depends on BOTH current z and current e)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate
%
% Naive QH dual pass over the DC argmax axis:
%   Valt/Policy4alt maximise  F + beta*EV        (the exponential value)
%   V/Policy4       maximise  F + beta0*beta*EV  (the QH-perceived value)
% Each maximisation is a full divide-and-conquer pass (its own level1/maxgap/level2).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
%
% lowmemory ladder (0-3) implemented.

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem((1:1:N_d2)',N_d1,1);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy4=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray');
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy4alt=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothzind=shiftdim((0:1:N_bothz-1),-1);

% Preallocate per-d3 (alt=exponential, tilde=QH-perceived)
V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Terminal: pure return, single DC pass. No continuation => V=Valt, Policy4=Policy4alt.
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        dind=rem(maxindex2-1,N_d)+1;
        d12_ind=rem(dind-1,N_d12)+1;
        Policy4alt(1,curraindex,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
        Policy4alt(2,curraindex,:,:,N_j)=ceil(d12_ind/N_d1);
        Policy4alt(3,curraindex,:,:,N_j)=ceil(dind/N_d12);
        Policy4alt(4,curraindex,:,:,N_j)=ceil(maxindex2/N_d);

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                eind=shiftdim((0:1:N_e-1),-2);
                allind=dind+N_d*a2ind+N_d*N_a2*bothzind+N_d*N_a2*N_bothz*eind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,curraindex,:,:,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,curraindex,:,:,N_j)=ceil(dind/N_d12);
                Policy4alt(4,curraindex,:,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,n_bothz,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                eind=shiftdim((0:1:N_e-1),-2);
                allind=dind+N_d*a2ind+N_d*N_a2*bothzind+N_d*N_a2*N_bothz*eind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,curraindex,:,:,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,curraindex,:,:,N_j)=ceil(dind/N_d12);
                Policy4alt(4,curraindex,:,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            dind=rem(maxindex2-1,N_d)+1;
            d12_ind=rem(dind-1,N_d12)+1;
            Policy4alt(1,curraindex,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy4alt(2,curraindex,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy4alt(3,curraindex,:,e_c,N_j)=ceil(dind/N_d12);
            Policy4alt(4,curraindex,:,e_c,N_j)=ceil(maxindex2/N_d);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d*a2ind+N_d*N_a2*shiftdim((0:1:N_bothz-1),-1);
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                    Policy4alt(2,curraindex,:,e_c,N_j)=ceil(d12_ind/N_d1);
                    Policy4alt(3,curraindex,:,e_c,N_j)=ceil(dind/N_d12);
                    Policy4alt(4,curraindex,:,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d*a2ind+N_d*N_a2*shiftdim((0:1:N_bothz-1),-1);
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy4alt(1,curraindex,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                    Policy4alt(2,curraindex,:,e_c,N_j)=ceil(d12_ind/N_d1);
                    Policy4alt(3,curraindex,:,e_c,N_j)=ceil(dind/N_d12);
                    Policy4alt(4,curraindex,:,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                Valt(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                dind=rem(maxindex2-1,N_d)+1;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,curraindex,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d12);
                Policy4alt(4,curraindex,semizblock,e_c,N_j)=ceil(maxindex2/N_d);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d*a2ind+N_d*N_a2*shiftdim((0:1:N_semiz-1),-1);
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4alt(1,curraindex,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                        Policy4alt(2,curraindex,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                        Policy4alt(3,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d12);
                        Policy4alt(4,curraindex,semizblock,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d*a2ind+N_d*N_a2*shiftdim((0:1:N_semiz-1),-1);
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4alt(1,curraindex,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                        Policy4alt(2,curraindex,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                        Policy4alt(3,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d12);
                        Policy4alt(4,curraindex,semizblock,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                dind=rem(maxindex2-1,N_d)+1;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12);
                Policy4alt(4,curraindex,z_c,e_c,N_j)=ceil(maxindex2/N_d);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d*a2ind;
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                        Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                        Policy4alt(3,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12);
                        Policy4alt(4,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Valt(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d*a2ind;
                        d12_ind=rem(dind-1,N_d12)+1;
                        Policy4alt(1,curraindex,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                        Policy4alt(2,curraindex,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                        Policy4alt(3,curraindex,z_c,e_c,N_j)=ceil(dind/N_d12);
                        Policy4alt(4,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                    end
                end
            end
        end
    end
    % Terminal period: no continuation, so QH-perceived value equals exponential value
    V(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy4(:,:,:,:,N_j)=Policy4alt(:,:,:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current_z, current_e); independent of d3, semiz
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % [N_d2,N_a2,N_z,N_e]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs_d2a1a2ze=repmat(a2primeProbs,N_a1,1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1); % [N_d2*N_a1,N_a2,N_bothz,N_e]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2ze,1,1,N_semiz,1);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            entireEV=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*entireEV;       % exponential
            DiscountedEV_tilde=beta0beta*entireEV; % QH-perceived

            % Level1 return matrix (shared by both passes)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            %% alt pass (exponential: F + beta*EV)
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimeze),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimeze),[N_d12,level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimeze),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimeze),[N_d12,level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_alt_e,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_alt_e(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_alt_e(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_tilde_e,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_tilde_e(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_tilde_e(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_alt_zb=DiscountedEV_alt(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_alt_zbe=DiscountedEV_alt_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_alt_zbe,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_zbe(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_zbe(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_tilde_zb=DiscountedEV_tilde(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_tilde_zbe=DiscountedEV_tilde_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_tilde_zbe,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_zbe(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_zbe(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_z(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_z(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policy4alt(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy4alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1);
    Policy4alt(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12);

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy4(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1);
    Policy4(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12);
end

%% Iterate backwards through j
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs_d2a1a2ze=repmat(a2primeProbs,N_a1,1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz,1);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz,1);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2ze,1,1,N_semiz,1);

    % Continuation value is the exponential value (Valt), integrated over e'
    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            entireEV=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_alt=beta*entireEV;
            DiscountedEV_tilde=beta0beta*entireEV;

            % Level1 return matrix (shared by both passes)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            %% alt pass (exponential: F + beta*EV)
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimeze),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_alt(d2aprimeze),[N_d12,level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_alt(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimeze),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV_tilde(d2aprimeze),[N_d12,level1iidiff(ii)*N_a2,N_bothz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind+N_d1*N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_tilde(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_alt_e=DiscountedEV_alt(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_alt_e,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_alt_e(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_alt_e(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_alt(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+repelem(DiscountedEV_tilde_e,N_d1,1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_tilde_e(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_tilde_e(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_tilde(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_alt_zb=DiscountedEV_alt(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_alt_zbe=DiscountedEV_alt_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_alt_zbe,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_zbe(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_zbe(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_alt(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_tilde_zb=DiscountedEV_tilde(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_tilde_zbe=DiscountedEV_tilde_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_tilde_zbe,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_zbe(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_zbe(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_semiz]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_tilde(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            %% alt pass (exponential: F + beta*EV)
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_alt_z=DiscountedEV_alt(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_alt_z,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_z(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_alt_z(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_alt(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end

            %% tilde pass (QH-perceived: F + beta0*beta*EV)
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_tilde_z=DiscountedEV_tilde(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+repelem(DiscountedEV_tilde_z,N_d1,1,1,1,1);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_tilde_z(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d1*N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d1*N_d2*a2ind;
                            Policy_ford3_tilde(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policy4alt(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy4alt(2,:,:,:,jj)=ceil(d12_ind/N_d1);
    Policy4alt(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12);

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,jj)=V_jj;
    Policy4(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy4(2,:,:,:,jj)=ceil(d12_ind/N_d1);
    Policy4(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
