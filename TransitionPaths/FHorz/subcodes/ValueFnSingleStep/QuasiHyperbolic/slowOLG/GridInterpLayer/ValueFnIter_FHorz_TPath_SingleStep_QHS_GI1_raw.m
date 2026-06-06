function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar. Policy is the QH (beta0*beta) choice; V output is Vunderbar at this period.
% Vhat is the agent's-perspective value (beta0*beta) at the QH policy.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');
Vhat=zeros(N_a,N_z,N_j,'gpuArray');

%%
if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_raw')
end

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
    [~,maxindex]=max(ReturnMatrix,[],2);

    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

    d_ind = rem(maxindexL2-1,N_d)+1;
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind + N_d*n2long*aind + N_d*n2long*N_a*zind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,N_j)=shiftdim(Vtempii,1);
    allind=d_ind+N_d*aind+N_d*N_a*zind;
    Policy(1,:,:,N_j)=d_ind;
    Policy(2,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix_z,[],2);

        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

        d_ind = rem(maxindexL2-1,N_d)+1;
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind + N_d*n2long*aind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,z_c,N_j)=shiftdim(Vtempii,1);
        allind=d_ind+N_d*aind;
        Policy(1,:,z_c,N_j)=d_ind;
        Policy(2,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    end

end

Vhat(:,:,N_j)=V(:,:,N_j);

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

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        % --- Vhat search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
        [~,maxindex]=max(entireRHS,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zBind;
        EVfine=reshape(EVinterp(aprimez(:)),[N_d*n2long,N_a,N_z]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

        d_ind = rem(maxindexL2-1,N_d)+1;
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind + N_d*n2long*aind + N_d*n2long*N_a*zind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        allind=d_ind+N_d*aind+N_d*N_a*zind;
        Policy(1,:,:,jj)=d_ind;
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);

        linidx=reshape(maxindexL2,[1,N_a*N_z])+N_d*n2long*(0:N_a*N_z-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
        Vhat(:,:,jj)=shiftdim(Vtempii,1);
        V(:,:,jj)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

            % --- Vhat search (beta0*beta) ---
            entireRHS_z=ReturnMatrix_z+beta0beta*shiftdim(EV_z,-1);
            [~,maxindex]=max(entireRHS_z,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2_z=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_z,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            EVfine_z=reshape(EVinterp_z(aprimeindexes(:)),[N_d*n2long,N_a]);
            entireRHS_L2_z=ReturnMatrix_L2_z+beta0beta*EVfine_z;
            [Vtempii,maxindexL2]=max(entireRHS_L2_z,[],1);

            d_ind = rem(maxindexL2-1,N_d)+1;
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (ReturnMatrix_L2_z(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2_z(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            allind=d_ind+N_d*aind;
            Policy(1,:,z_c,jj)=d_ind;
            Policy(2,:,z_c,jj)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(3,:,z_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1);

            linidx_z=reshape(maxindexL2,[1,N_a])+N_d*n2long*(0:N_a-1);
            EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
            Vhat(:,z_c,jj)=shiftdim(Vtempii,1);
            V(:,z_c,jj)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end


% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1);
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust;
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];


end
