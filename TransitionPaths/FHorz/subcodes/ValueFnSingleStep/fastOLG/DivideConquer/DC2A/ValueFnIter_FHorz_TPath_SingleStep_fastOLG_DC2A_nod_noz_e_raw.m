function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_noz_e_raw(V,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% DC2A (no d, no z, has e): divide-and-conquer in the first endogenous state
%       (a1), iterate (vectorize) over the second endogenous state (a2).
% fastOLG layout (noz_e): V is (a*j)-by-e, Policy is (1, a, j, e)
% e is i.i.d., so EV is integrated over e' using pi_e_J before the DC pass.
% pi_e_J is (a*j, e) for fastOLG; e_gridvals_J is (j, N_e, l_e) on input.

N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_j,N_e,'gpuArray'); % joint (a1prime,a2prime) index at each (a,j,e) cell

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

% Pre-shift e_gridvals_J so the fastOLG DC2A_nod return-fn helper (treating e
% as its 'z') can index it directly: helper expects [1,1,1,1,N_j,N_e,l_e].
e_gridvals_J=shiftdim(e_gridvals_J,-4); % [1,1,1,1,N_j,N_e,l_e]

% precompute indices (match non-e plain DC2A_nod_noz_raw)
a2ind=gpuArray(0:1:N_a2-1);             % [1,N_a2]    a2prime offset (dim 2) and a2 stride in allind
jind =shiftdim(gpuArray(0:1:N_j-1),-1); % [1,1,N_j]   age offset in allind (dim 3 of policy decode)
jBind=shiftdim(gpuArray(0:1:N_j-1),-3); % [1,1,1,1,N_j] age offset in aprimej (dim 5 of EV)


%% Create the big 'next period (of transition path) expected value fn',
% integrating out the i.i.d. e shock.

ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j);

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

if vfoptions.EVpre==0
    % V is (N_a*N_j, N_e); e is i.i.d., integrate it out using pi_e_J.
    EVpre=[sum(V(N_a+1:end,:).*pi_e_J(1:end-N_a,:),2); zeros(N_a,1,'gpuArray')]; % zeros at j=N_j (terminal age has no continuation in TPath)
    EV=reshape(EVpre,[N_a1,N_a2,1,1,N_j]); % (a1prime,a2prime,1,1,j)
elseif vfoptions.EVpre==1
    % 'Matched Expectations Path': input V is already E[V'|.] across e'
    EV=reshape(V,[N_a1,N_a2,1,1,N_j]);
end
V=zeros(N_a,N_j,N_e,'gpuArray'); % V is over (a,j,e)

DiscountedEV=reshape(DiscountFactor_J,[1,1,1,1,N_j]).*EV; % [N_a1,N_a2,1,1,N_j]; trailing N_e broadcasts in via the per-e loop


%% Loop over e (lowmemory-style; mirrors GI2A_nod_noz_e_raw lowmemory==1 branch)
special_n_e=ones(1,length(n_e));
for e_c=1:N_e
    e_vals=e_gridvals_J(1,1,1,1,:,e_c,:); % [1,1,1,1,N_j,1,l_e]

    % n-Monotonicity (Level 1 coarse)
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_e, N_j, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_vals, ReturnFnParamsAgeMatrix,1);
    % shape: [N_a1, N_a2, level1n, N_a2, N_j] (trailing N_z=1 dropped)

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (1, a2prime, a1=level1ii, a2, j)
    [~,maxindex1]=max(entireRHS_ii,[],1);
    % maxindex1 shape: [1, N_a2, level1n, N_a2, N_j]

    % Joint (a1prime,a2prime) at the level1 anchor points
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_j]),[],1);
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
    V(curraindex,:,e_c)=shiftdim(Vtempii,1);
    Policy(curraindex,:,e_c)=shiftdim(maxindex2,1);

    % Improved version
    maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii));
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-N_j
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-by-n_a2-by-1-by-n_a2-by-N_j
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_e, N_j, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsAgeMatrix,2);
            aprimej=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*jBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimej,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_j]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,e_c)=shiftdim(Vtempii,1);
            a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
            a2primeind=ceil(maxindex/(maxgap(ii)+1));
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind;
            Policy(curraindex,:,e_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        else
            loweredge=maxindex1(1,:,ii,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, special_n_e, N_j, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_vals, ReturnFnParamsAgeMatrix,2);
            aprimej=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*jBind;
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimej,[1*N_a2,level1iidiff(ii)*N_a2,N_j]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,e_c)=shiftdim(Vtempii,1);
            a1primeind=1;
            a2primeind=maxindex;
            maxindexfix=a1primeind+N_a1*(a2primeind-1);
            allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*jind;
            Policy(curraindex,:,e_c)=shiftdim(maxindexfix+loweredge(allind)-1,1);
        end
    end
end


%% fastOLG with e, so V is (a*j)-by-e
V=reshape(V,[N_a*N_j,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
