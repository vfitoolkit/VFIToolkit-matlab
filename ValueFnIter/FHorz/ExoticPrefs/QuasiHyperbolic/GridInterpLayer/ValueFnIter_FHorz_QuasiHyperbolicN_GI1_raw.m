function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI_raw.
% Has d variables. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{d,a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{d,a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

aind=gpuArray(0:1:N_a-1);                 % 1-by-N_a
zind=shiftdim(gpuArray(0:1:N_z-1),-1);    % 1-by-1-by-N_z
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);   % 1-by-1-by-N_z (alias for broadcasting EVinterp)

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
            [~,maxindex]=max(ReturnMatrix,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end

    Vtilde=V;

else
    % Using V_Jplus1 (V for naive)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_z,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_V=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindex_V]=max(entireRHS_V,[],2);
        midpoint_V=max(min(maxindex_V,n_a-1),2);
        aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez_V=aprimeindexes_V+n2aprime*zBind;
        entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimez_V(:)),[N_d*n2long,N_a,N_z]);
        [Vtempii,~]=max(entireRHS_L2_V,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0*beta) ---
        entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
        midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
        aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez_Vt=aprimeindexes_Vt+n2aprime*zBind;
        entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimez_Vt(:)),[N_d*n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind;
        Policy(1,:,:,N_j)=d_ind;
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_V=ReturnMatrix+beta*shiftdim(EV_z,-1);
            [~,maxindex_V]=max(entireRHS_V,[],2);
            midpoint_V=max(min(maxindex_V,n_a-1),2);
            aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp_z(aprimeindexes_V(:)),[N_d*n2long,N_a]);
            [Vtempii,~]=max(entireRHS_L2_V,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0*beta) ---
            entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV_z,-1);
            [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
            midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
            aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp_z(aprimeindexes_Vt(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
            Vtilde(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,z_c,N_j)=d_ind;
            Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
            Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
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
    EV=EVsource.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        % --- V search (beta) ---
        entireRHS_V=ReturnMatrix+beta*shiftdim(EV,-1);
        [~,maxindex_V]=max(entireRHS_V,[],2);
        midpoint_V=max(min(maxindex_V,n_a-1),2);
        aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez_V=aprimeindexes_V+n2aprime*zBind;
        entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp(aprimez_V(:)),[N_d*n2long,N_a,N_z]);
        [Vtempii,~]=max(entireRHS_L2_V,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0*beta) ---
        entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
        midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
        aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez_Vt=aprimeindexes_Vt+n2aprime*zBind;
        entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp(aprimez_Vt(:)),[N_d*n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zind;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

            % --- V search (beta) ---
            entireRHS_V=ReturnMatrix+beta*shiftdim(EV_z,-1);
            [~,maxindex_V]=max(entireRHS_V,[],2);
            midpoint_V=max(min(maxindex_V,n_a-1),2);
            aprimeindexes_V=(midpoint_V+(midpoint_V-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2_V=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes_V),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2_V=ReturnMatrix_L2_V+beta*reshape(EVinterp_z(aprimeindexes_V(:)),[N_d*n2long,N_a]);
            [Vtempii,~]=max(entireRHS_L2_V,[],1);
            V(:,z_c,jj)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0*beta) ---
            entireRHS_Vt=ReturnMatrix+beta0beta*shiftdim(EV_z,-1);
            [~,maxindex_Vt]=max(entireRHS_Vt,[],2);
            midpoint_Vt=max(min(maxindex_Vt,n_a-1),2);
            aprimeindexes_Vt=(midpoint_Vt+(midpoint_Vt-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2_Vt=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes_Vt),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2_Vt=ReturnMatrix_L2_Vt+beta0beta*reshape(EVinterp_z(aprimeindexes_Vt(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2_Vt,[],1);
            Vtilde(:,z_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            Policy(1,:,z_c,jj)=d_ind;
            Policy(2,:,z_c,jj)=shiftdim(squeeze(midpoint_Vt(allind)),-1);
            Policy(3,:,z_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
end

%% Post-process Policy: convert [d_ind, midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(3,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:,:)+N_d*(Policy(2,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
