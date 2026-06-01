function [V, Policy, Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% fastOLG just means parallelize over "age" (j)
% V carries Vunderbar for Sophisticated QH
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime
Vhat=zeros(N_a,N_j,'gpuArray'); % beta0*beta-step value (snapshot of V before Vunderbar transform)

%%
a_grid=gpuArray(a_grid);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_aprime by N_a*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);
Beta0_J=CreateAgeMatrixFromParams(Parameters, {vfoptions.QHadditionaldiscount},N_j);
Beta0DiscountFactor_J=Beta0_J.*DiscountFactor_J;
BetaMinusBeta0Beta_J=DiscountFactor_J-Beta0DiscountFactor_J;

if vfoptions.EVpre==0
    EV=zeros(N_a,N_j,'gpuArray');
    EV(:,1:N_j-1)=V(:,2:end);
    EV=reshape(EV,[N_a,1,N_j]);
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    EV=reshape(V,[N_a,1,N_j]); % input V is of size [N_a,N_j] and we want to use the whole thing
end
V=zeros(N_a,N_j,'gpuArray'); % V is over (a,j); for Sophisticated QH carries Vunderbar

Beta0DiscountedEV=reshape(Beta0DiscountFactor_J,[1,1,N_j]).*EV; % beta0_j*beta_j*EV

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid, a_grid(level1ii), ReturnFnParamsAgeMatrix,1);

entireRHS_ii=ReturnMatrix_ii+Beta0DiscountedEV; % (aprime,a and j), autofills a for expectation term

[Vtempii,maxindex1]=max(entireRHS_ii,[],1);

V(level1ii,:)=shiftdim(Vtempii,1);
Policy(level1ii,:)=shiftdim(maxindex1,1);

maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
for ii=1:(vfoptions.level1n-1)
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
        aprimeindexes=loweredge+(0:1:maxgap(ii))'; % ' due to no d
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=aprimeindexes+N_a*shiftdim((0:1:N_j-1),-1);
        entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+loweredge-1,1);
    else
        loweredge=maxindex1(1,ii,:);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_fastOLG_Disc_DC1_nod_noz(ReturnFn, N_j, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsAgeMatrix,2);
        aprime=loweredge+N_a*shiftdim((0:1:N_j-1),-1);
        entireRHS_ii=ReturnMatrix_ii_dc+Beta0DiscountedEV(aprime);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+loweredge-1,1);
    end
end

%% Re-evaluate V at Policy with beta (not beta0*beta): V=Vunderbar=Vhat+(beta-beta0*beta)*EV_at_policy
Vhat=V; % snapshot Vhat before Vunderbar transform
EV_2d=reshape(EV,[N_a,N_j]); % (aprime,j)
EV_at_policy=EV_2d(Policy+N_a*(0:1:N_j-1));
V=V+reshape(BetaMinusBeta0Beta_J,[1,N_j]).*EV_at_policy;

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
