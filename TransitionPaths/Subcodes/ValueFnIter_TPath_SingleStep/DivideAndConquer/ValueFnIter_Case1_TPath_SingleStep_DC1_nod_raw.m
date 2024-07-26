function [V, Policy]=ValueFnIter_Case1_TPath_SingleStep_DC1_nod_raw(Vnext,n_a,n_z, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
EV=sum(EV,2); % sum over z', leaving a singular second dimension
DiscountedEV=DiscountFactorParamsVec*EV;

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals, ReturnFnParamsVec,1);

entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

%Calc the max and it's index
[Vtempii,maxindex1]=max(entireRHS_ii,[],1);

V(level1ii,:)=shiftdim(Vtempii,1);
Policy(level1ii,:)=shiftdim(maxindex1,1);

% Attempt for improved version
maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
for ii=1:(vfoptions.level1n-1)
    if maxgap(ii)>0
        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is 1-by-1-by-n_z
        aprimeindexes=loweredge+(0:1:maxgap(ii))';
        % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
        aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-1); % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(aprimez); % autofill level1iidiff(ii) in 2nd dimension
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(Vtempii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(maxindex+loweredge-1,1);
    else
        loweredge=maxindex1(1,ii,:);
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
        aprimez=loweredge+N_a*shiftdim((0:1:N_z-1),-1); % the current aprimeii(ii):aprimeii(ii+1)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(aprimez); % autofill level1iidiff(ii) in 2nd dimension
        % can skip mmax() over as just a single point
        V(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(entireRHS_ii,1);
        Policy(level1ii(ii)+1:level1ii(ii+1)-1,:)=shiftdim(ones(size(maxindex1))+loweredge-1,1);
    end
end


end
