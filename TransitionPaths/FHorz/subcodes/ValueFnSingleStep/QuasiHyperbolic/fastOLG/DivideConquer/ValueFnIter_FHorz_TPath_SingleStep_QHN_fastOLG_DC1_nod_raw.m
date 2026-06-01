function [V, Policy, Policyalt, Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z; for Naive QH the input/output V carries Valt (exp-discounter value)
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG

N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime
Policyalt=zeros(N_a,N_j,N_z,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_j,N_z,'gpuArray'); % beta0*beta-step value (max at final level2 of Policy sweep)

z_gridvals_J=shiftdim(z_gridvals_J,-2); % [1,1,N_j,N_z,l_z]

%%
% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

jind=shiftdim(gpuArray(0:1:N_j-1),-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
Beta0_J=CreateAgeMatrixFromParams(Parameters, {vfoptions.QHadditionaldiscount},N_j);
Beta0DiscountFactor_J=Beta0_J.*DiscountFactor_J;

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
    EV=reshape(V,[N_a,1,N_j,N_z]).*shiftdim(pi_z_J,-2); % input V is already of size [N_a*N_j,N_z] and we want to use the whole thing.
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end
V=zeros(N_a,N_j,N_z,'gpuArray'); % preallocate: V is over (a,j,z); for Naive QH carries Valt

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % beta_j*EV
Beta0DiscountedEV=reshape(Beta0DiscountFactor_J,[1,1,N_j]).*EV; % beta0_j*beta_j*EV

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid, a_grid(level1ii), z_gridvals_J, ReturnFnParamsAgeMatrix,1);

    %% Valt (beta): write V (=Valt) and Policyalt
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % (aprime,a and j,z), autofills a for expectation term

    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    % Store
    V(level1ii,:,:)=Vtempii;
    Policyalt(level1ii,:,:)=maxindex1; % aprime

    maxgap_V=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii))'; % ' due to no d
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(aprimejz);
            [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindexalt+loweredge-1,1); % loweredge
        else
            loweredge=maxindex1(1,ii,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=loweredge+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(aprimejz);
            [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
            Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindexalt+loweredge-1,1); % loweredge
        end
    end

    %% Policy (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV;

    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    Policy(level1ii,:,:)=maxindex1;
    Vtilde(level1ii,:,:)=Vtempii;

    maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=aprimeindexes+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprimejz);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+loweredge-1,1);
            Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
        else
            loweredge=maxindex1(1,ii,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, n_z, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=loweredge+N_a*jind+N_a*N_j*zind;
            entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprimejz);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(maxindex+loweredge-1,1);
            Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:)=shiftdim(Vtempii,1);
        end
    end

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        Beta0DiscountedEV_z=Beta0DiscountedEV(:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid, a_grid(level1ii), z_vals, ReturnFnParamsAgeMatrix,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

        V(level1ii,:,z_c)=Vtempii;
        Policyalt(level1ii,:,z_c)=maxindex1;

        maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=aprimeindexes+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(aprimej);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindexalt+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=loweredge+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(aprimej);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
                Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindexalt+loweredge-1,1);
            end
        end

        %% Policy (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV_z;

        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

        Policy(level1ii,:,z_c)=maxindex1;
        Vtilde(level1ii,:,z_c)=Vtempii;

        maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=aprimeindexes+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV_z(aprimej);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+loweredge-1,1);
                Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1(1,ii,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod(ReturnFn, special_n_z, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=loweredge+N_a*jind;
                entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV_z(aprimej);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(maxindex+loweredge-1,1);
                Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c)=shiftdim(Vtempii,1);
            end
        end
    end
end

%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z]);
Vtilde=reshape(Vtilde,[N_a*N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
Policyalt=shiftdim(Policyalt,-1);


end
