function [V,Policy2]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% DC2A: divide-and-conquer in the first endogenous state only (a1), iterate
%       (vectorize) over the second endogenous state (a2).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,'gpuArray'); % first dim indexes the optimal choice for d and (a1prime,a2prime)

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% pre-shift z_gridvals_J for the DC2A fastOLG return-fn helper
z_gridvals_J=shiftdim(z_gridvals_J,-5); % [1,1,1,1,1,N_j,N_z,l_z]

%%
% n-Monotonicity (a1 only)
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute indices
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1; sits at dim 3 (a2prime) of a1primeindexes
jind=shiftdim(gpuArray(0:1:N_j-1),-1); % sits at dim 3 (N_j) of maxindex, used in allind
jBind=shiftdim(gpuArray(0:1:N_j-1),-4); % sits at dim 6 (N_j) of a1primeindexes, used in aprimejz
zind=shiftdim(gpuArray(0:1:N_z-1),-2); % sits at dim 4 (N_z) of maxindex, used in allind
zBind=shiftdim(gpuArray(0:1:N_z-1),-5); % sits at dim 7 (N_z) of a1primeindexes, used in aprimejz


%% First, create the big 'next period (of transition path) expected value fn'
% spanning all ages simultaneously.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    EVpre=zeros(N_a1,N_a2,1,1,N_j,N_z);
    EVpre(:,:,1,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a1,N_a2,1,1,N_j-1,N_z]); % zeros in j=N_j so pi_z_J just produces zeros there
    EV=EVpre.*shiftdim(pi_z_J,-4); % pi_z_J is [N_j,N_z',N_z] -> [1,1,1,1,N_j,N_z',N_z]
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]); % sum over z'
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expectations Path'
    EV=reshape(V,[N_a1,N_a2,1,1,N_j,N_z]).*shiftdim(pi_z_J,-4);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]);
end
V=zeros(N_a,N_j,N_z,'gpuArray'); % V is over (a,j,z)

DiscountedEV=shiftdim(reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV,-1); % [1,N_a1,N_a2,1,1,N_j,N_z] — 1st dim autofills d


if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a2prime,a1,a2,j,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,a1prime,a2prime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_j,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,:)=shiftdim(Vtempii,1);
    Policy(curraindex,:,:)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % a1prime possibilities are n_d-by-maxgap(ii)+1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1', but needs to be after N_a1
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1; % already includes -1
            a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind+N_d*N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1)+N_a1*a2Bind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            dind=(rem(maxindex-1,N_d)+1);
            a1primeind=0; %1-1; already includes -1
            a2primeind=ceil(maxindex/N_d)-1; % already includes -1
            maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
            allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind+N_d*N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
        end
    end

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,1,1,:,z_c,:); % shape [1,1,1,1,1,N_j,1,l_z]
        DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        % First, we want a1prime conditional on (d,1,a2prime,a1,a2,j)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,a1prime,a2prime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_j]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,z_c)=shiftdim(Vtempii,1);
        Policy(curraindex,:,z_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-N_j
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a1_grid(a1primeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=repelem(a1primeindexes,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*jBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprimej,[N_d*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a1primeind=rem(ceil(maxindex/N_d)-1,maxgap(ii)+1)+1-1;
                a2primeind=ceil(maxindex/(N_d*(maxgap(ii)+1)))-1;
                maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
                allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind;
                Policy(curraindex,:,z_c)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, special_n_z, N_j, d_gridvals, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=repelem(loweredge,1,1,1,level1iidiff(ii),1,1)+N_a1*a2Bind+N_a*jBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprimej,[N_d*1*N_a2,level1iidiff(ii)*N_a2,N_j]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,z_c)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a1primeind=0;
                a2primeind=ceil(maxindex/N_d)-1;
                maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind;
                allind=dind+N_d*a2primeind+N_d*N_a2*repelem(a2ind,1,level1iidiff(ii))+N_d*N_a2*N_a2*jind;
                Policy(curraindex,:,z_c)=shiftdim(maxindexfix+N_d*(loweredge(allind)-1),1);
            end
        end
    end
end

%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point

end
