function [Vhat,Policy3,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_DC1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic + ExperienceAssete + SemiExo, Divide-and-Conquer (DC1 over a1prime). No d1.
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (optional), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; not on z or semiz)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate
%
% Sophisticated QH over the DC argmax axis:
%   Policy3 (and Vhat) come from the  F + beta0*beta*EV  argmax (QH-perceived).
%   Vunderbar is the  F + beta*EV  RHS GATHERED at that same DC argmax (NOT re-maximised).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Vunderbar.
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothzind=shiftdim((0:1:N_bothz-1),-1);

% Preallocate per-d3 (hat=QH-perceived argmax, under=beta-RHS gathered at that argmax)
V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;



%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Terminal: pure return, single DC pass. No continuation => Vunderbar=Vhat.
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d23*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d23)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                eind=shiftdim((0:1:N_e-1),-2);
                allind=dind+N_d23*a2ind+N_d23*N_a2*bothzind+N_d23*N_a2*N_bothz*eind;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d23)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                eind=shiftdim((0:1:N_e-1),-2);
                allind=dind+N_d23*a2ind+N_d23*N_a2*bothzind+N_d23*N_a2*N_bothz*eind;
                Policy3(1,curraindex,:,:,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,:,:,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,:,:,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d23*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d23*a2ind+N_d23*N_a2*shiftdim((0:1:N_bothz-1),-1);
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Vhat(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d23)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d23*a2ind+N_d23*N_a2*shiftdim((0:1:N_bothz-1),-1);
                    Policy3(1,curraindex,:,e_c,N_j)=rem(dind-1,N_d2)+1;
                    Policy3(2,curraindex,:,e_c,N_j)=ceil(dind/N_d2);
                    Policy3(3,curraindex,:,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d23*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                Vhat(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                dind=rem(maxindex2-1,N_d23)+1;
                Policy3(1,curraindex,semizblock,e_c,N_j)=rem(dind-1,N_d2)+1;
                Policy3(2,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d2);
                Policy3(3,curraindex,semizblock,e_c,N_j)=ceil(maxindex2/N_d23);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2ind+N_d23*N_a2*shiftdim((0:1:N_semiz-1),-1);
                        Policy3(1,curraindex,semizblock,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,semizblock,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,semizblock,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2ind+N_d23*N_a2*shiftdim((0:1:N_semiz-1),-1);
                        Policy3(1,curraindex,semizblock,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,semizblock,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,semizblock,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d23*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
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
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2ind;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d23)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d23*a2ind;
                        Policy3(1,curraindex,z_c,e_c,N_j)=rem(dind-1,N_d2)+1;
                        Policy3(2,curraindex,z_c,e_c,N_j)=ceil(dind/N_d2);
                        Policy3(3,curraindex,z_c,e_c,N_j)=ceil(maxindex/N_d23+loweredge(allind)-1);
                    end
                end
            end
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current_z, current_e); independent of d3, semiz
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEV=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_under=beta*entireEV;      % exponential
            DiscountedEV_hat=beta0beta*entireEV;   % QH-perceived

            % Level1: argmax on hat, gather under at that argmax
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            entireRHS_hat=ReturnMatrix_ii_d3+DiscountedEV_hat;
            [~,maxindex1]=max(entireRHS_hat,[],2);
            entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under=ReturnMatrix_ii_d3+DiscountedEV_under;
            entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]);
            M=vfoptions.level1n*N_a2;
            maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_bothz-1),-1) + (N_d2*N_a1)*M*N_bothz*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under_flat(maxindexfull);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
            V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
            Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    firstdim=N_d2*(maxgap(ii)+1);
                    Mblock=level1iidiff(ii)*N_a2;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1) + firstdim*Mblock*N_bothz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d2*a2ind+N_d2*N_a2*bothzind+N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    firstdim=N_d2;
                    Mblock=level1iidiff(ii)*N_a2;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1) + firstdim*Mblock*N_bothz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d2*a2ind+N_d2*N_a2*bothzind+N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,e_c);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                entireRHS_hat=ReturnMatrix_ii_d3e+DiscountedEV_hat_e;
                [~,maxindex1]=max(entireRHS_hat,[],2);
                entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]);
                [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                entireRHS_under=ReturnMatrix_ii_d3e+DiscountedEV_under_e;
                entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]);
                M=vfoptions.level1n*N_a2;
                maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_bothz-1),-1);
                Vtempii_under=entireRHS_under_flat(maxindexfull);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        firstdim=N_d2*(maxgap(ii)+1);
                        Mblock=level1iidiff(ii)*N_a2;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3e+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        firstdim=N_d2;
                        Mblock=level1iidiff(ii)*N_a2;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3e+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_under_zb=DiscountedEV_under(:,:,:,:,semizblock,:);
                DiscountedEV_hat_zb=DiscountedEV_hat(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_under_zbe=DiscountedEV_under_zb(:,:,:,:,:,e_c);
                    DiscountedEV_hat_zbe=DiscountedEV_hat_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_hat=ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe;
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe;
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]);
                    M=vfoptions.level1n*N_a2;
                    maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_semiz-1),-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            firstdim=N_d2*(maxgap(ii)+1);
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            firstdim=N_d2;
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    DiscountedEV_under_z=DiscountedEV_under(:,:,:,:,z_c,e_c);
                    DiscountedEV_hat_z=DiscountedEV_hat(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_hat=ReturnMatrix_ii_d3ze+DiscountedEV_hat_z;
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3ze+DiscountedEV_under_z;
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2]);
                    M=vfoptions.level1n*N_a2;
                    maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            firstdim=N_d2*(maxgap(ii)+1);
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_z(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_z(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            firstdim=N_d2;
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_z(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_z(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=Vhat_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(d3maxindex,-1);
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_bothz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2);
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3maxindex_lin-1)),[N_a,N_bothz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    % Continuation value is Vunderbar, integrated over e'
    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEV=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_under=beta*entireEV;
            DiscountedEV_hat=beta0beta*entireEV;

            % Level1: argmax on hat, gather under at that argmax
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            entireRHS_hat=ReturnMatrix_ii_d3+DiscountedEV_hat;
            [~,maxindex1]=max(entireRHS_hat,[],2);
            entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under=ReturnMatrix_ii_d3+DiscountedEV_under;
            entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz,N_e]);
            M=vfoptions.level1n*N_a2;
            maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_bothz-1),-1) + (N_d2*N_a1)*M*N_bothz*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under_flat(maxindexfull);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
            V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
            Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    firstdim=N_d2*(maxgap(ii)+1);
                    Mblock=level1iidiff(ii)*N_a2;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1) + firstdim*Mblock*N_bothz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d2*a2ind+N_d2*N_a2*bothzind+N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3)+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    firstdim=N_d2;
                    Mblock=level1iidiff(ii)*N_a2;
                    entireRHS_hat=reshape(ReturnMatrix_ii_d3+DiscountedEV_hat(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEV_under(d2aprimeze),[firstdim,Mblock,N_bothz,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1) + firstdim*Mblock*N_bothz*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under(maxindexfull);
                    V_ford3_hat(curraindex,:,:,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,:,:,d3_c)=shiftdim(Vtempii_under,1);
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    eind=shiftdim((0:1:N_e-1),-2);
                    allind=dind+N_d2*a2ind+N_d2*N_a2*bothzind+N_d2*N_a2*N_bothz*eind;
                    Policy_ford3_hat(curraindex,:,:,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,e_c);
                DiscountedEV_hat_e=DiscountedEV_hat(:,:,:,:,:,e_c);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

                entireRHS_hat=ReturnMatrix_ii_d3e+DiscountedEV_hat_e;
                [~,maxindex1]=max(entireRHS_hat,[],2);
                entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]);
                [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                entireRHS_under=ReturnMatrix_ii_d3e+DiscountedEV_under_e;
                entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]);
                M=vfoptions.level1n*N_a2;
                maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_bothz-1),-1);
                Vtempii_under=entireRHS_under_flat(maxindexfull);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        firstdim=N_d2*(maxgap(ii)+1);
                        Mblock=level1iidiff(ii)*N_a2;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3e+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                        firstdim=N_d2;
                        Mblock=level1iidiff(ii)*N_a2;
                        entireRHS_hat=reshape(ReturnMatrix_ii_d3e+DiscountedEV_hat_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEV_under_e(d2aprimez),[firstdim,Mblock,N_bothz]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_bothz-1),-1);
                        Vtempii_under=entireRHS_under(maxindexfull);
                        V_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                        V_ford3_under(curraindex,:,e_c,d3_c)=shiftdim(Vtempii_under,1);
                        dind=(rem(maxindex-1,N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_bothz-1),-1);
                        Policy_ford3_hat(curraindex,:,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_under_zb=DiscountedEV_under(:,:,:,:,semizblock,:);
                DiscountedEV_hat_zb=DiscountedEV_hat(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_under_zbe=DiscountedEV_under_zb(:,:,:,:,:,e_c);
                    DiscountedEV_hat_zbe=DiscountedEV_hat_zb(:,:,:,:,:,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_hat=ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe;
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe;
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_semiz]);
                    M=vfoptions.level1n*N_a2;
                    maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1) + (N_d2*N_a1)*M*shiftdim((0:N_semiz-1),-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            firstdim=N_d2*(maxgap(ii)+1);
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_semiz-1),-3);
                            firstdim=N_d2;
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_zbe(d2aprimez),[firstdim,Mblock,N_semiz]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_semiz-1),-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,semizblock,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind+N_d2*N_a2*shiftdim((0:1:N_semiz-1),-1);
                            Policy_ford3_hat(curraindex,semizblock,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            DiscountedEV_under=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            DiscountedEV_hat=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);

            % Single pass: argmax on hat (F+beta0*beta*EV), gather under (F+beta*EV) at that argmax
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    DiscountedEV_under_z=DiscountedEV_under(:,:,:,:,z_c,e_c);
                    DiscountedEV_hat_z=DiscountedEV_hat(:,:,:,:,z_c,e_c);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    entireRHS_hat=ReturnMatrix_ii_d3ze+DiscountedEV_hat_z;
                    [~,maxindex1]=max(entireRHS_hat,[],2);
                    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1,vfoptions.level1n*N_a2]);
                    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                    entireRHS_under=ReturnMatrix_ii_d3ze+DiscountedEV_under_z;
                    entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1,vfoptions.level1n*N_a2]);
                    M=vfoptions.level1n*N_a2;
                    maxindexfull=maxindex2 + (N_d2*N_a1)*(0:M-1);
                    Vtempii_under=entireRHS_under_flat(maxindexfull);

                    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                    V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                    V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                    Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex2,1);

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            firstdim=N_d2*(maxgap(ii)+1);
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_z(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_z(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                            firstdim=N_d2;
                            Mblock=level1iidiff(ii)*N_a2;
                            entireRHS_hat=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_hat_z(d2aprime),[firstdim,Mblock]);
                            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEV_under_z(d2aprime),[firstdim,Mblock]);
                            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                            maxindexfull=maxindex + firstdim*(0:Mblock-1);
                            Vtempii_under=entireRHS_under(maxindexfull);
                            V_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_hat,1);
                            V_ford3_under(curraindex,z_c,e_c,d3_c)=shiftdim(Vtempii_under,1);
                            dind=(rem(maxindex-1,N_d2)+1);
                            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                            allind=dind+N_d2*a2ind;
                            Policy_ford3_hat(curraindex,z_c,e_c,d3_c)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                        end
                    end
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=Vhat_jj;
    Policy3(2,:,:,:,jj)=shiftdim(d3maxindex,-1);
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_bothz*N_e,1]);
    d2a1prime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2);
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3maxindex_lin-1)),[N_a,N_bothz,N_e]);
end


end
