function [Vtilde,Policy2,V]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_noz_raw.
% Has d variables. No z variable. No e variable. GPU (parallel==2 only).
%
% Naive: V_j = max u + beta*E[V_{j+1}]  (used as EVsource for next period)
%        Vtilde_j = max u + beta_0*beta*E[V_{j+1}]  (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Vtilde=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex2,1);
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec,2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        end
    end
    Vtilde=V;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    maxgap_V=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimeindexes),[N_d*(maxgap_V(ii)+1),1]);
            [Vtempii,~]=max(entireRHS_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(loweredge);
            [Vtempii,~]=max(entireRHS_ii,[],1);
            V(curraindex,N_j)=shiftdim(Vtempii,1);
        end
    end

    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    Vtilde(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex2,1);
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vtilde(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(loweredge);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vtilde(curraindex,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        end
    end
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

    EV=V(:,jj+1);

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,~]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    V(level1ii,jj)=shiftdim(Vtempii,1);
    maxgap_V=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*reshape(EV(aprimeindexes),[N_d*(maxgap_V(ii)+1),1]);
            [Vtempii,~]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(loweredge);
            [Vtempii,~]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
        end
    end

    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    Vtilde(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex2,1);
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vtilde(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(loweredge);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Vtilde(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        end
    end
end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray');
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end
