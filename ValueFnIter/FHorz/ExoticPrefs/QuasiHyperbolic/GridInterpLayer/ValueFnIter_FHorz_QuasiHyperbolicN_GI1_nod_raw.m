function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI_nod_raw.
% No d variables. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]          (time-consistent)
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}] (agent's actual choice)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); % [midpoint; aprimeL2ind]

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1); % 1-by-1-by-N_z

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [~,maxindex]=max(ReturnMatrix,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            [~,maxindex]=max(ReturnMatrix,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
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
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        % --- V search (beta) ---
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,~]=max(entireRHS_L2,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            % --- V search (beta) ---
            entireRHS=ReturnMatrix+beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,~]=max(entireRHS_L2,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0*beta) ---
            entireRHS=ReturnMatrix+beta0beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vtilde(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % --- V search (beta) ---
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,~]=max(entireRHS_L2,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        % --- Vtilde search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            % --- V search (beta) ---
            entireRHS=ReturnMatrix+beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,~]=max(entireRHS_L2,[],1);
            V(:,z_c,jj)=shiftdim(Vtempii,1);
            % --- Vtilde search (beta0*beta) ---
            entireRHS=ReturnMatrix+beta0beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vtilde(:,z_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,jj)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,jj)=shiftdim(maxindexL2,-1);
        end
    end
end

%% Post-process Policy: convert [midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:,:)+N_a*(Policy(2,:,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
