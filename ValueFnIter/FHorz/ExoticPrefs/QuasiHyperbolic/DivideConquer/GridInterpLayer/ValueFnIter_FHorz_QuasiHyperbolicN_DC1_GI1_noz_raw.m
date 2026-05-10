function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_GI1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI_noz_raw.
% Has d variables. No z variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{d,a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{d,a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]

midpoints_jj=zeros(N_d,1,N_a,'gpuArray');

aind=gpuArray(0:1:N_a-1);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1;
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            [~,maxindex]=max(ReturnMatrix_ii,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    Vtilde=V;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    % --- V search (beta) ---
    entireRHS_ii_V=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1_V]=max(entireRHS_ii_V,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1_V;
    maxgap_V=max(maxindex1_V(:,1,2:end)-maxindex1_V(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1_V(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_V(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii_V,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1_V(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2_V,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    % --- Vtilde search (beta0beta) ---
    entireRHS_ii_Vt=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1_Vt]=max(entireRHS_ii_Vt,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1_Vt;
    maxgap_Vt=max(maxindex1_Vt(:,1,2:end)-maxindex1_Vt(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_Vt(ii)>0
            loweredge=min(maxindex1_Vt(:,1,ii),n_a-maxgap_Vt(ii));
            aprimeindexes=loweredge+(0:1:maxgap_Vt(ii));
            ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_Vt(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii_Vt,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1_Vt(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
    Vtilde(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
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
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta;

    EVsource=V(:,jj+1);
    EV=EVsource;
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    % --- V search (beta) ---
    entireRHS_ii_V=ReturnMatrix_ii+beta*shiftdim(EV,-1);
    [~,maxindex1_V]=max(entireRHS_ii_V,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1_V;
    maxgap_V=max(maxindex1_V(:,1,2:end)-maxindex1_V(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_V(ii)>0
            loweredge=min(maxindex1_V(:,1,ii),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii));
            ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_V(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii_V,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1_V(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2_V,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    % --- Vtilde search (beta0beta) ---
    entireRHS_ii_Vt=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1_Vt]=max(entireRHS_ii_Vt,[],2);
    midpoints_jj(:,1,level1ii)=maxindex1_Vt;
    maxgap_Vt=max(maxindex1_Vt(:,1,2:end)-maxindex1_Vt(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap_Vt(ii)>0
            loweredge=min(maxindex1_Vt(:,1,ii),n_a-maxgap_Vt(ii));
            aprimeindexes=loweredge+(0:1:maxgap_Vt(ii));
            ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec,3);
            entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV(reshape(aprimeindexes(:),[N_d,(maxgap_Vt(ii)+1),1]));
            [~,maxindex]=max(entireRHS_ii_Vt,[],2);
            midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1_Vt(:,1,ii);
            midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
        end
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
    Vtilde(:,jj)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    allind=d_ind+N_d*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1);
    Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
end

%% Post-process Policy: convert [d_ind, midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(3,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:)+N_d*(Policy(2,:,:)-1)+N_d*N_a*(Policy(3,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
