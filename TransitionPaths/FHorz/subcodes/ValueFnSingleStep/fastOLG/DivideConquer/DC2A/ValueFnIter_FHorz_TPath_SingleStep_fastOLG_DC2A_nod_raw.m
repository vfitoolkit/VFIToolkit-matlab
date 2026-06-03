function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% DC2A: divide-and-conquer in the first endogenous state only (a1), iterate
%       (vectorize) over the second endogenous state (a2).

N_a=prod(n_a);
N_z=prod(n_z);

% fastOLG, so a-j-z
Policy=zeros(N_a,N_j,N_z,'gpuArray'); % joint (a1prime,a2prime) index at each (a,j,z) cell

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% pre-shift z_gridvals_J for the DC2A_nod fastOLG return-fn helper
z_gridvals_J=shiftdim(z_gridvals_J,-4); % [1,1,1,1,N_j,N_z,l_z]

%%
% n-Monotonicity (a1 only)
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute indices
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
jind=shiftdim(gpuArray(0:1:N_j-1),-1); % sits at dim 3 (N_j) of maxindex, used in allind
jBind=shiftdim(gpuArray(0:1:N_j-1),-3); % sits at dim 5 (N_j) of aprimeindexes, used in aprimejz
zind=shiftdim(gpuArray(0:1:N_z-1),-2); % sits at dim 4 (N_z) of maxindex, used in allind
zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % sits at dim 6 (N_z) of aprimeindexes, used in aprimejz


%% First, create the big 'next period (of transition path) expected value fn'
% spanning all ages simultaneously.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    EVpre=zeros(N_a1,N_a2,1,1,N_j,N_z);
    EVpre(:,:,1,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a1,N_a2,1,1,N_j-1,N_z]); % zeros in j=N_j
    EV=EVpre.*shiftdim(pi_z_J,-4); % [1,1,1,1,N_j,N_z',N_z]
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]);
elseif vfoptions.EVpre==1
    EV=reshape(V,[N_a1,N_a2,1,1,N_j,N_z]).*shiftdim(pi_z_J,-4);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]);
end
V=zeros(N_a,N_j,N_z,'gpuArray'); % preallocate: V is over (a,j,z)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1,N_a2,1,1,N_j,N_z]


if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,1);

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (1,a2prime,a1,a2,j,z)
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Now, get and store the full (a1prime,a2prime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_j,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,:)=shiftdim(Vtempii,1);
    Policy(curraindex,:,:)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii));
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
            a2primeind=ceil(maxindex/(maxgap(ii)+1));
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind+N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        else
            loweredge=maxindex1(1,:,ii,:,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[1*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            a1primeind=1;
            a2primeind=maxindex;
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind+N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        end
    end

elseif vfoptions.lowmemory==1

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,1,:,z_c,:); % shape [1,1,1,1,N_j,1,l_z]
        DiscountedEV_z=DiscountedEV(:,:,:,:,:,z_c);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsAgeMatrix,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

        [~,maxindex1]=max(entireRHS_ii,[],1);

        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_j]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,z_c)=shiftdim(Vtempii,1);
        Policy(curraindex,:,z_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii));
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-N_j
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*jBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprimej,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,z_c)=shiftdim(Vtempii,1);
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1);
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind;
                Policy(curraindex,:,z_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_z, N_j, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsAgeMatrix,2);
                aprimej=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*jBind;
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(reshape(aprimej,[1*N_a2,level1iidiff(ii)*N_a2,N_j]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,z_c)=shiftdim(Vtempii,1);
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1);
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind;
                Policy(curraindex,:,z_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end
    end
end

%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
