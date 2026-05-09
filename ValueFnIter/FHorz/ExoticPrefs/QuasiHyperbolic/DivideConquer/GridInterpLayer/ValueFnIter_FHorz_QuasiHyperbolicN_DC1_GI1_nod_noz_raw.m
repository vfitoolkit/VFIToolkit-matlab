function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_GI1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI_nod_noz_raw.
% No d variables. No z variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(2,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind]

midpoints_jj=zeros(1,N_a,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    [~,maxindex]=max(ReturnMatrix_ii,[],1);
    midpoints_jj(1,level1ii)=maxindex;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        [~,maxindex]=max(ReturnMatrix_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);

    Vtilde=V;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    % --- V search (beta) ---
    entireRHS_ii_V=ReturnMatrix_ii+beta*EV;
    [~,maxindex_V]=max(entireRHS_ii_V,[],1);
    midpoints_jj(1,level1ii)=maxindex_V;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_V,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2_V,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    % --- Vtilde search (beta0beta) ---
    entireRHS_ii_Vt=ReturnMatrix_ii+beta0beta*EV;
    [~,maxindex_Vt]=max(entireRHS_ii_Vt,[],1);
    midpoints_jj(1,level1ii)=maxindex_Vt;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_Vt,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
    Vtilde(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);
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

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    % --- V search (beta) ---
    entireRHS_ii_V=ReturnMatrix_ii+beta*EV;
    [~,maxindex_V]=max(entireRHS_ii_V,[],1);
    midpoints_jj(1,level1ii)=maxindex_V;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_V=ReturnMatrix_ii_V+beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_V,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2_V,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    % --- Vtilde search (beta0beta) ---
    entireRHS_ii_Vt=ReturnMatrix_ii+beta0beta*EV;
    [~,maxindex_Vt]=max(entireRHS_ii_Vt,[],1);
    midpoints_jj(1,level1ii)=maxindex_Vt;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_Vt,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
    Vtilde(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);
end

%% Post-process Policy: convert [midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:)+N_a*(Policy(2,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
