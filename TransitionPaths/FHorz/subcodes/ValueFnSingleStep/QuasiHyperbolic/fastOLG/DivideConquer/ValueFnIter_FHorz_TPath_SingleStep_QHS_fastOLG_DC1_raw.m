function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z; for Sophisticated QH the input/output V carries Vunderbar
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for d and aprime
Vhat=zeros(N_a,N_j,N_z,'gpuArray'); % beta0*beta-step value (snapshot of V before Vunderbar transform)

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
zind=shiftdim(gpuArray(0:1:N_z-1),-3);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-2);
jCind=gpuArray(0:1:N_j-1); % 2D [1,N_j]; for EV_at_policy lookup where aprime_ind is 2D/3D [N_a,N_j(,N_z)]
zCind=shiftdim(gpuArray(0:1:N_z-1),-1); % 3D [1,1,N_z]; pairs with jCind for the same lookup


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
Beta0_J=CreateAgeMatrixFromParams(Parameters, {vfoptions.QHadditionaldiscount},N_j);
Beta0DiscountFactor_J=Beta0_J.*DiscountFactor_J;
BetaMinusBeta0Beta_J=DiscountFactor_J-Beta0DiscountFactor_J; % (beta - beta0*beta) per age

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=zeros(N_a,1,N_j,N_z);
    EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2); % input V is already of size [N_a,N_j,N_z] and we want to use the whole thing
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end
V=zeros(N_a,N_j,N_z,'gpuArray'); % V is over (a,j,z); for Sophisticated QH carries Vunderbar

Beta0DiscountedEV=shiftdim(reshape(Beta0DiscountFactor_J,[1,1,N_j]).*EV,-1); % [1,N_a,1,N_j,N_z] beta0_j*beta_j*EV


if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV; % (d,aprime,a and j,z), autofills a for expectation term

    % First, we want aprime conditional on (d,1,a,j)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_z-1),-1)); % aprime

    % Store
    V(level1ii,:,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=aprimeindexes+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV(aprimez(:)),[N_d*(maxgap(ii)+1),1,N_j,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind+N_d*N_j*zBind)-1),1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_z, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=loweredge+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV(aprimez(:)),[N_d,1,N_j,N_z]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind+N_d*N_j*zBind)-1),1);
        end
    end

    %% Re-evaluate V at Policy with beta (not beta0*beta): V=Vunderbar=Vhat+(beta-beta0*beta)*EV_at_policy
    Vhat=V; % snapshot Vhat before Vunderbar transform
    aprime_ind=ceil(Policy/N_d);
    EV_at_policy=reshape(EV,[N_a,N_j,N_z]); % (aprime,j,z)
    EV_at_policy=EV_at_policy(aprime_ind+N_a*jCind+N_a*N_j*zCind);
    V=V+reshape(BetaMinusBeta0Beta_J,[1,N_j,1]).*EV_at_policy;

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,N_j,N_z,l_z) for fastOLG
        Beta0DiscountedEV_z=Beta0DiscountedEV(:,:,:,:,z_c);
        EV_z=reshape(EV(:,:,:,z_c),[N_a,N_j]); % (aprime,j)

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid, a_grid(level1ii), z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV_z; % (d,aprime,a and j,z), autofills j for expectation term

        % First, we want aprime conditional on (d,1,a,j)
        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
        % Now, we get the d and we store the (d,aprime) and the

        %Calc the max and it's index
        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        maxindex2=shiftdim(maxindex2,2); % d
        maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)); % aprime

        % Store
        V(level1ii,:,z_c)=shiftdim(Vtempii,2);
        Policy(level1ii,:,z_c)=maxindex2+N_d*(maxindex1d-1); % d,aprime

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprime=aprimeindexes+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV_z(aprime(:)),[N_d*(maxgap(ii)+1),1,N_j]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprime=loweredge+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV_z(aprime(:)),[N_d,1,N_j]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+N_d*(loweredge(rem(maxindex-1,N_d)+1+N_d*jBind)-1),1);
            end
        end

        %% Re-evaluate V at Policy with beta (not beta0*beta) for this z
        Vhat(:,:,z_c)=V(:,:,z_c); % snapshot Vhat before Vunderbar transform (for this z)
        aprime_ind_z=ceil(Policy(:,:,z_c)/N_d);
        EV_at_policy_z=EV_z(aprime_ind_z+N_a*jCind);
        V(:,:,z_c)=V(:,:,z_c)+reshape(BetaMinusBeta0Beta_J,[1,N_j]).*EV_at_policy_z;
    end
end

%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z]);
Vhat=reshape(Vhat,[N_a*N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point



end
