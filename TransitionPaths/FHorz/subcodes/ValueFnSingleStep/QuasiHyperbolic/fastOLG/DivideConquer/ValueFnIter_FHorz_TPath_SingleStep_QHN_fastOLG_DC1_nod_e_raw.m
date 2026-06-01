function [V, Policy, Policyalt, Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e; for Naive QH carries Valt

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % first dim indexes the optimal choice for aprime
Policyalt=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % beta0*beta-step value (max at final level2 of Policy sweep)

z_gridvals_J=shiftdim(z_gridvals_J,-2); % needed shape for ReturnFnMatrix with fastOLG and DC1
e_gridvals_J=reshape(e_gridvals_J,[1,1,N_j,1,N_e,length(n_e)]); % needed shape for ReturnFnMatrix with fastOLG and DC1

%%

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% First, create the big 'next period (of transition path) expected value fn.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
Beta0_J=CreateAgeMatrixFromParams(Parameters, {vfoptions.QHadditionaldiscount},N_j);
Beta0DiscountFactor_J=Beta0_J.*DiscountFactor_J;

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
    EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EVpre=[reshape(V,[N_a*N_j,N_z,N_e].*pi_e_J,3)];  % input V is already of size [N_a,N_j] and we want to use the whole thing
    EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end
V=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % V is over (a,j,z,e); for Naive QH carries Valt

DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % beta_j*EV
Beta0DiscountedEV=reshape(Beta0DiscountFactor_J,[1,1,N_j]).*EV; % beta0_j*beta_j*EV

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid, a_grid(level1ii), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,1);

    %% Valt (beta): write V (=Valt) and Policyalt
    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % (aprime,a and j,z,e), autofills a and e for expectation term

    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    V(level1ii,:,:,:)=Vtempii;
    Policyalt(level1ii,:,:,:)=maxindex1;

    maxgap_V=squeeze(max(max(max(maxindex1(1,2:end,:,:,:)-maxindex1(1,1:end-1,:,:,:),[],5),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        if maxgap_V(ii)>0
            loweredge=min(maxindex1(1,ii,:,:,:),n_a-maxgap_V(ii));
            aprimeindexes=loweredge+(0:1:maxgap_V(ii))'; % ' due to no d
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(aprimez);
            [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindexalt+loweredge-1,1);
        else
            loweredge=maxindex1(1,ii,:,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=loweredge+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(aprimez);
            [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
            V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
            Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindexalt+loweredge-1,1);
        end
    end

    %% Policy (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV;

    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

    Policy(level1ii,:,:,:)=maxindex1;
    Vtilde(level1ii,:,:,:)=Vtempii;

    maxgap=squeeze(max(max(max(maxindex1(1,2:end,:,:,:)-maxindex1(1,1:end-1,:,:,:),[],5),[],4),[],3));
    for ii=1:(vfoptions.level1n-1)
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprimez);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+loweredge-1,1);
            Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
        else
            loweredge=maxindex1(1,ii,:,:,:);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimez=loweredge+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
            entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprimez);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(maxindex+loweredge-1,1);
            Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:,:)=shiftdim(Vtempii,1);
        end
    end

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,j,1,prod(n_e),l_e)

        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, N_j, a_grid, a_grid(level1ii), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV;

        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

        V(level1ii,:,:,e_c)=Vtempii;
        Policyalt(level1ii,:,:,e_c)=maxindex1;

        maxgap_V=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
        for ii=1:(vfoptions.level1n-1)
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap_V(ii));
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
                entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV(aprimez);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindexalt+loweredge-1,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=loweredge+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
                entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV(aprimez);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                V(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
                Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindexalt+loweredge-1,1);
            end
        end

        %% Policy (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii_e+Beta0DiscountedEV;

        [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

        Policy(level1ii,:,:,e_c)=maxindex1;
        Vtilde(level1ii,:,:,e_c)=Vtempii;

        maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
        for ii=1:(vfoptions.level1n-1)
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
                entireRHS_ii=ReturnMatrix_ii_e_dc+Beta0DiscountedEV(aprimez);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+loweredge-1,1);
                Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
            else
                loweredge=maxindex1(1,ii,:,:);
                ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
                aprimez=loweredge+N_a*shiftdim((0:1:N_j-1),-1)+N_a*N_j*shiftdim((0:1:N_z-1),-2);
                entireRHS_ii=ReturnMatrix_ii_e_dc+Beta0DiscountedEV(aprimez);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(maxindex+loweredge-1,1);
                Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,:,e_c)=shiftdim(Vtempii,1);
            end
        end
    end
elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,:,z_c,:); % z_gridvals_J has shape (1,1,j,prod(n_z),l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);
        Beta0DiscountedEV_z=Beta0DiscountedEV(:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,j,1,prod(n_e),l_e)

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, N_j, a_grid, a_grid(level1ii), z_vals, e_vals, ReturnFnParamsAgeMatrix,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            V(level1ii,:,z_c,e_c)=Vtempii;
            Policyalt(level1ii,:,z_c,e_c)=maxindex1;

            maxgap_V=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
            for ii=1:(vfoptions.level1n-1)
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap_V(ii));
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(aprime);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindexalt+loweredge-1,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=loweredge+N_a*shiftdim((0:1:N_j-1),-1);
                    entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_z(aprime);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                    Policyalt(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindexalt+loweredge-1,1);
                end
            end

            %% Policy (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV_z;

            [Vtempii,maxindex1]=max(entireRHS_ii,[],1);

            Policy(level1ii,:,z_c,e_c)=maxindex1;
            Vtilde(level1ii,:,z_c,e_c)=Vtempii;

            maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
            for ii=1:(vfoptions.level1n-1)
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1);
                    entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV_z(aprime);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+loweredge-1,1);
                    Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                else
                    loweredge=maxindex1(1,ii,:);
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_vals, e_vals, ReturnFnParamsAgeMatrix,2);
                    aprime=loweredge+N_a*shiftdim((0:1:N_j-1),-1);
                    entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV_z(aprime);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Policy(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(maxindex+loweredge-1,1);
                    Vtilde(level1ii(ii)+1:level1ii(ii+1)-1,:,z_c,e_c)=shiftdim(Vtempii,1);
                end
            end
        end
     end
end

%% fastOLG with z & e, so need output to take certain shapes
V=reshape(V,[N_a*N_j,N_z,N_e]);
Vtilde=reshape(Vtilde,[N_a*N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point
Policyalt=shiftdim(Policyalt,-1);


end
