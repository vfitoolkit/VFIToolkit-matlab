function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_noz_e_raw(n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI_noz_e_raw.
% Has d variables. No z variable. Has e variables. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{d,a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{d,a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray');

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

aind=gpuArray(0:1:N_a-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-1);

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

pi_e_J=shiftdim(pi_e_J,-1);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,e_c,N_j)=d_ind;
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
    Vtilde=V;
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % --- V search (beta) ---
        entireRHS_V=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindex_V]=max(entireRHS_V,[],2);
        midpoint_V=max(min(maxindex_V,n_a-1),2);
        aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EVinterp(aprimeindexes_V(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,~]=max(entireRHS_ii_V,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0beta) ---
        entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
        midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
        aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EVinterp(aprimeindexes_Vt(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_Vt,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);
            % --- V search (beta) ---
            entireRHS_V_e=ReturnMatrix_e+beta*shiftdim(EV,-1);
            [~,maxindex_V]=max(entireRHS_V_e,[],2);
            midpoint_V=max(min(maxindex_V,n_a-1),2);
            aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_V_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,e_val,ReturnFnParamsVec,2);
            entireRHS_ii_V_e=ReturnMatrix_ii_V_e+beta*reshape(EVinterp(aprimeindexes_V(:)),[N_d*n2long,N_a]);
            [Vtempii,~]=max(entireRHS_ii_V_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0beta) ---
            entireRHS_Vt_e=ReturnMatrix_e+beta0beta*shiftdim(EV,-1);
            [~,maxindex_Vt]=max(entireRHS_Vt_e,[],2);
            midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
            aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_Vt_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,e_val,ReturnFnParamsVec,2);
            entireRHS_ii_Vt_e=ReturnMatrix_ii_Vt_e+beta0beta*reshape(EVinterp(aprimeindexes_Vt(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_Vt_e,[],1);
            Vtilde(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,e_c,N_j)=d_ind;
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
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

    EVsource=V(:,:,jj+1);
    EV=sum(EVsource.*pi_e_J(1,:,jj),2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        % --- V search (beta) ---
        entireRHS_V=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindex_V]=max(entireRHS_V,[],2);
        midpoint_V=max(min(maxindex_V,n_a-1),2);
        aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        entireRHS_ii_V=ReturnMatrix_ii_V+beta*reshape(EVinterp(aprimeindexes_V(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,~]=max(entireRHS_ii_V,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0beta) ---
        entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
        midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
        aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        entireRHS_ii_Vt=ReturnMatrix_ii_Vt+beta0beta*reshape(EVinterp(aprimeindexes_Vt(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii_Vt,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1);
            % --- V search (beta) ---
            entireRHS_V_e=ReturnMatrix_e+beta*shiftdim(EV,-1);
            [~,maxindex_V]=max(entireRHS_V_e,[],2);
            midpoint_V=max(min(maxindex_V,n_a-1),2);
            aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_V_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,e_val,ReturnFnParamsVec,2);
            entireRHS_ii_V_e=ReturnMatrix_ii_V_e+beta*reshape(EVinterp(aprimeindexes_V(:)),[N_d*n2long,N_a]);
            [Vtempii,~]=max(entireRHS_ii_V_e,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0beta) ---
            entireRHS_Vt_e=ReturnMatrix_e+beta0beta*shiftdim(EV,-1);
            [~,maxindex_Vt]=max(entireRHS_Vt_e,[],2);
            midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
            aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_Vt_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,e_val,ReturnFnParamsVec,2);
            entireRHS_ii_Vt_e=ReturnMatrix_ii_Vt_e+beta0beta*reshape(EVinterp(aprimeindexes_Vt(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_Vt_e,[],1);
            Vtilde(:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,e_c,jj)=d_ind;
            Policy(2,:,e_c,jj)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
            Policy(3,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
end

%% Post-process Policy: convert [d_ind, midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(3,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1);

Policy=Policy(1,:,:,:)+N_d*(Policy(2,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:)-1);

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
