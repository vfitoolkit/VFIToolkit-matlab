function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,e), rather than standard (a,z,e)
% V is (a,j)-by-e; for Sophisticated QH carries Vunderbar

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

% fastOLG, so a-j-e
Policy=zeros(N_a,N_j,N_e,'gpuArray'); % first dim indexes the optimal choice for d and aprime
Vhat=zeros(N_a,N_j,N_e,'gpuArray'); % beta0*beta-step value (snapshot of V before Vunderbar transform)

% e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,N_e,length(n_e)]); % needed shape for ReturnFnMatrix with fastOLG and DC1

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-2);
jBind=shiftdim(gpuArray(0:1:N_j-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
jCind=gpuArray(0:1:N_j-1); % 2D [1,N_j]; for EV_at_policy lookup

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_e (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
Beta0_J=CreateAgeMatrixFromParams(Parameters, {vfoptions.QHadditionaldiscount},N_j);
Beta0DiscountFactor_J=Beta0_J.*DiscountFactor_J;
BetaMinusBeta0Beta_J=DiscountFactor_J-Beta0DiscountFactor_J;

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% pi_e_J is (a,j)-by-e
if vfoptions.EVpre==0
    EV=[sum(V(N_a+1:end,:).*pi_e_J(1:end-N_a,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=sum(reshape(V,[N_a*N_j,N_e]).*pi_e_J,2);  % input V is already of size [N_a,N_j] and we want to use the whole thing
end
V=zeros(N_a,N_j,N_e,'gpuArray'); % V is over (a,j,e); for Sophisticated QH carries Vunderbar

Beta0DiscountedEV=shiftdim(reshape(Beta0DiscountFactor_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]),-1); % [1,aprime,1,j] beta0_j*beta_j*EV

if vfoptions.lowmemory==0


    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV; % (d,aprime,a and j,e), autofills a for expectation term

    % First, we want aprime conditional on (d,1,a,j,e)
    [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);
    % Now, we get the d and we store the (d,aprime) and the

    %Calc the max and it's index
    [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
    maxindex2=shiftdim(maxindex2,2); % d
    maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1)+N_d*vfoptions.level1n*N_j*shiftdim((0:1:N_e-1),-1)); % aprime

    % Store
    V(level1ii,:,:)=shiftdim(Vtempii,2);
    Policy(level1ii,:,:)=maxindex2+N_d*(maxindex1d-1); % d,aprime

    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprime=aprimeindexes+N_a*jind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV(aprime(:)),[N_d*(maxgap(ii)+1),1,N_j,N_e]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,ii,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, n_e, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprime=loweredge+N_a*jind;
            entireRHS_ii=ReturnMatrix_ii_dc+reshape(Beta0DiscountedEV(aprime(:)),[N_d,1,N_j,N_e]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            d_ind=rem(maxindex-1,N_d)+1;
            allind=d_ind+N_d*jBind+N_d*N_j*eBind;
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
        end
    end

    %% Re-evaluate V at Policy with beta (not beta0*beta): V=Vunderbar=Vhat+(beta-beta0*beta)*EV_at_policy
    Vhat=V; % snapshot Vhat before Vunderbar transform
    aprime_ind=ceil(Policy/N_d); % [N_a,N_j,N_e]
    EV_2d=reshape(EV,[N_a,N_j]); % (aprime,j); e broadcasts
    EV_at_policy=EV_2d(aprime_ind+N_a*jCind); % [N_a,N_j,N_e]
    V=V+reshape(BetaMinusBeta0Beta_J,[1,N_j,1]).*EV_at_policy;

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:); % e_gridvals_J has shape (1,1,1,j,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid, a_grid(level1ii), e_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii_e+Beta0DiscountedEV;

        [RMtemp_ii,maxindex1]=max(entireRHS_ii,[],2);

        [Vtempii,maxindex2]=max(RMtemp_ii,[],1);
        maxindex2=shiftdim(maxindex2,2);
        maxindex1d=maxindex1(maxindex2+N_d*(0:1:vfoptions.level1n-1)'+N_d*vfoptions.level1n*(0:1:N_j-1));

        V(level1ii,:,e_c)=shiftdim(Vtempii,2);
        Policy(level1ii,:,e_c)=maxindex2+N_d*(maxindex1d-1);

        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_vals, ReturnFnParamsAgeMatrix,2);
                aprime=aprimeindexes+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_e_dc+reshape(Beta0DiscountedEV(aprime(:)),[N_d*(maxgap(ii)+1),1,N_j]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1(ReturnFn, n_d, special_n_e, N_j, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_vals, ReturnFnParamsAgeMatrix,2);
                aprime=loweredge+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_e_dc+reshape(Beta0DiscountedEV(aprime(:)),[N_d,1,N_j]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindex-1,N_d)+1;
                allind=d_ind+N_d*jBind;
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            end
        end

        %% Re-evaluate V at Policy with beta (not beta0*beta) for this e
        Vhat(:,:,e_c)=V(:,:,e_c); % snapshot Vhat before Vunderbar transform (for this e)
        aprime_ind_e=ceil(Policy(:,:,e_c)/N_d);
        EV_2d=reshape(EV,[N_a,N_j]);
        EV_at_policy_e=EV_2d(aprime_ind_e+N_a*jCind);
        V(:,:,e_c)=V(:,:,e_c)+reshape(BetaMinusBeta0Beta_J,[1,N_j]).*EV_at_policy_e;
    end
end

%% fastOLG with e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_e]);
Vhat=reshape(Vhat,[N_a*N_j,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
