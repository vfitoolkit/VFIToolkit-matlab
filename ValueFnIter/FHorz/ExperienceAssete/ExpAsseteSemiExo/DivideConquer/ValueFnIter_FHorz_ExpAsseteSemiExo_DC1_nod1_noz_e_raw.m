function [V,Policy3]=ValueFnIter_FHorz_ExpAsseteSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% semiz is semi-exog state, e is i.i.d. start-of-period (required); no z, no d1
% aprimeFn = aprimeFn(d2, a2, e, ...)

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory==0
    eBind=shiftdim((0:1:N_e-1),-2);
    eind=shiftdim((0:1:N_e-1),-4);
else
    special_n_e=ones(1,length(n_e));
end

if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
else
    semizind=shiftdim((0:1:N_semiz-1),-3);
    semizBind=shiftdim((0:1:N_semiz-1),-1);
end

V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=shiftdim((0:1:N_a2-1),-2);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        dind=rem(maxindex2-1,N_d23)+1;
        Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1;
        Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2);
        Policy3(3,curraindex,:,:,N_j)=ceil(maxindex2/N_d23);

        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d23)+1);
                a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                allind=dind+N_d23*a2Bind+N_d23*N_a2*semizBind+N_d23*N_a2*N_semiz*eBind;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d23)+1);
                a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                allind=dind+N_d23*a2Bind+N_d23*N_a2*semizBind+N_d23*N_a2*N_semiz*eBind;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d23*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            dind=rem(maxindex2-1,N_d23)+1;
            Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1;
            Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2);
            Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex2/N_d23);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d23*a2Bind+N_d23*N_a2*semizBind;
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d23*a2Bind+N_d23*N_a2*semizBind;
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d23*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                dind=rem(maxindex2-1,N_d23)+1;
                Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex2/N_d23);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2Bind;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2Bind;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    end
                end
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind+N_d2*N_a*N_semiz*eind;
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind+N_d2*N_a2*N_semiz*eBind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind+N_d2*N_a*N_semiz*eind;
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind+N_d2*N_a2*N_semiz*eBind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEV_e,[N_d2,N_a1,1,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind;
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_e(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind;
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_e(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEV_ze,[N_d2,N_a1,1,N_a2]);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_ze(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2Bind;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_ze(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2Bind;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz,N_e]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind+N_d2*N_a*N_semiz*eind;
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind+N_d2*N_a2*N_semiz*eBind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind+N_d2*N_a*N_semiz*eind;
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3+DiscountedEV(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_semiz,N_e]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_jj(curraindex,:,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind+N_d2*N_a2*N_semiz*eBind;
                    Policy_ford3_jj(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_e=DiscountedEV(:,:,:,:,:,e_c);

                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_ii_d3e=ReturnMatrix_ii_d3e+reshape(DiscountedEV_e,[N_d2,N_a1,1,N_a2,N_semiz]);

                [~,maxindex1]=max(entireRHS_ii_d3e,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3e,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind;
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_e(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*semizind;
                        entireRHS_ii=reshape(ReturnMatrix_ii_d3e+DiscountedEV_e(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_semiz]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2Bind+N_d2*N_a2*semizBind;
                        Policy_ford3_jj(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_semiz,N_e]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_ze=DiscountedEV(:,:,:,:,z_c,e_c);

                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_ii_d3ze=ReturnMatrix_ii_d3ze+reshape(DiscountedEV_ze,[N_d2,N_a1,1,N_a2]);

                    [~,maxindex1]=max(entireRHS_ii_d3ze,[],2);
                    [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3ze,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_ze(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2Bind;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_ze(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                            V_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2Bind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2Bind;
                            Policy_ford3_jj(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2);
end


end
