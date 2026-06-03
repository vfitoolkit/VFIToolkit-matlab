function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_e_raw(V,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% DC2A (no d, has z+e): divide-and-conquer in the first endogenous state (a1),
%       iterate (vectorize) over the second endogenous state (a2).
% fastOLG layout: V is (a*j)-by-z-by-e, Policy is (1, a, j, z, e)
% pi_z_J is (j,z',z); pi_e_J is (a*j, z, e) for fastOLG (i.i.d. e, broadcast across a/j/z)
% z_gridvals_J is (j,N_z,l_z) and e_gridvals_J is (j,N_e,l_e) on input.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% fastOLG, so a-j-z-e
Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % joint (a1prime,a2prime) index at each (a,j,z,e) cell

%%
a_grid=gpuArray(a_grid);

n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=prod(n_a2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity (DC on a1 only)
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Pre-shift z/e grids so the fastOLG DC2A_nod_e return-fn helper can index directly:
% helper expects z at dim 6, e at dim 7, with l-component on dim 8.
z_gridvals_J=reshape(z_gridvals_J,[1,1,1,1,N_j,N_z,1,length(n_z)]); % [1,1,1,1,N_j,N_z,1,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,1,N_j,1,N_e,length(n_e)]); % [1,1,1,1,N_j,1,N_e,l_e]

% precompute indices (match non-e plain DC2A_nod_raw)
a2ind =gpuArray(0:1:N_a2-1);            % [1,N_a2]
jind  =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]
jBind =shiftdim(gpuArray(0:1:N_j-1),-3); % [1,1,1,1,N_j]
zind  =shiftdim(gpuArray(0:1:N_z-1),-2); % [1,1,1,N_z]
zBind =shiftdim(gpuArray(0:1:N_z-1),-4); % [1,1,1,1,1,N_z]


%% First, create the big 'next period (of transition path) expected value fn'.
% V is (N_a*N_j, N_z, N_e). Integrate over e' (i.i.d.) using pi_e_J, then over
% z' (Markov) using pi_z_J. The resulting EV depends on (a',j,z), not e.

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.EVpre==0
    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % (N_a*N_j, N_z); zeros at j=N_j (terminal)
    EVpre=reshape(EVpre,[N_a1,N_a2,1,1,N_j,N_z]);
    EV=EVpre.*shiftdim(pi_z_J,-4); % [1,1,1,1,N_j,N_z',N_z]
    EV(isnan(EV))=0; % -Inf*0 = NaN, replace with 0 (the 0 comes from transition prob)
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]);
elseif vfoptions.EVpre==1
    EV=reshape(V,[N_a1,N_a2,1,1,N_j,N_z]).*shiftdim(pi_z_J,-4);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,6),[N_a1,N_a2,1,1,N_j,N_z]);
end
V=zeros(N_a,N_j,N_z,N_e,'gpuArray'); % V is over (a,j,z,e)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1,N_a2,1,1,N_j,N_z]


%% Loop over e (lowmemory-style; z stays vectorized inside)
special_n_e=ones(1,length(n_e));
for e_c=1:N_e
    e_vals=e_gridvals_J(1,1,1,1,:,1,e_c,:); % [1,1,1,1,N_j,1,1,l_e]

    % n-Monotonicity (Level 1 coarse)
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,1);
    % shape: [N_a1, N_a2, level1n, N_a2, N_j, N_z] (trailing N_e=1 dropped)

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (1,a2prime,a1=level1ii,a2,j,z)
    [~,maxindex1]=max(entireRHS_ii,[],1);
    % maxindex1 shape: [1, N_a2, level1n, N_a2, N_j, N_z]

    % Joint (a1prime,a2prime) at the level1 anchor points
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_j,N_z]),[],1);
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
    Policy(curraindex,:,:,e_c)=shiftdim(maxindex2,1);

    % Improved version
    maxgap=squeeze(max(max(max(max(maxindex1(1,:,2:end,:,:,:)-maxindex1(1,:,1:end-1,:,:,:),[],6),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:,:),N_a1-maxgap(ii));
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-N_j-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, N_j, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
            a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
            a2primeind=ceil(maxindex/(maxgap(ii)+1));
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind+N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:,e_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        else
            loweredge=maxindex1(1,:,ii,:,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod_e(ReturnFn, n_z, special_n_e, N_j, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix,2);
            aprimejz=repelem(loweredge,1,1,level1iidiff(ii),1,1,1)+N_a1*a2ind+N_a*jBind+N_a*N_j*zBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimejz,[1*N_a2,level1iidiff(ii)*N_a2,N_j,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
            a1primeind=1;
            a2primeind=maxindex;
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind+N_a2*N_a2*N_j*zind;
            Policy(curraindex,:,:,e_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        end
    end
end


%% fastOLG with z and e, so V is (a*j)-by-z-by-e
V=reshape(V,[N_a*N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
