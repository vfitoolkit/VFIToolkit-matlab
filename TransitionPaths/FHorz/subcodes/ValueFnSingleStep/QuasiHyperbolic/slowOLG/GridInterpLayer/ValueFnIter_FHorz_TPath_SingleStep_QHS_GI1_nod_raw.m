function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar. Policy is QH choice (beta0*beta); V output is Vunderbar.
% Vhat is the agent's-perspective value (beta0*beta) at the QH policy.

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); % [midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray');
Vhat=zeros(N_a,N_z,N_j,'gpuArray');

%%
if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_nod_raw')
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1);

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
    [~,maxindex]=max(ReturnMatrix,[],1);

    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

    isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
        [~,maxindex]=max(ReturnMatrix,[],1);

        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

        isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V(:,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
    end
end

Vhat(:,:,N_j)=V(:,:,N_j);


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

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        % --- Vhat search (beta0*beta) ---
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        EVfine=reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Policy(1,:,:,jj)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1);

        linidx=reshape(maxindexL2,[1,N_a*N_z])+n2long*(0:N_a*N_z-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
        Vhat(:,:,jj)=shiftdim(Vtempii,1);
        V(:,:,jj)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            % --- Vhat search (beta0*beta) ---
            entireRHS_z=ReturnMatrix_z+beta0beta*EV_z;
            [~,maxindex]=max(entireRHS_z,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2_z=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            EVfine_z=reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            entireRHS_L2_z=ReturnMatrix_L2_z+beta0beta*EVfine_z;
            [Vtempii,maxindexL2]=max(entireRHS_L2_z,[],1);

            isInfLower    = (ReturnMatrix_L2_z(1,     :) == -Inf);
            isInfUpper    = (ReturnMatrix_L2_z(n2long,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,z_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Policy(1,:,z_c,jj)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,jj)=shiftdim(maxindexL2,-1);

            linidx_z=reshape(maxindexL2,[1,N_a])+n2long*(0:N_a-1);
            EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
            Vhat(:,z_c,jj)=shiftdim(Vtempii,1);
            V(:,z_c,jj)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end

% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];


end
