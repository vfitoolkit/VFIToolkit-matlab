function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_noz_raw(Vnext,n_a, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, divide-and-conquer (one endogenous state), no d, no z, no e.
% Vnext is [N_a,1]. No exogenous state to loop over, so vfoptions.lowmemory is ignored (everything at once).

N_a=prod(n_a);

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));

V=zeros(N_a,1,'gpuArray');
Policy=zeros(N_a,1,'gpuArray'); %first dim indexes the optimal choice for aprime

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% No shocks, so the expectation is just next period value fn
EV=Vnext; % [N_a,1], aprime-by-1
DiscountedEV=DiscountFactorParamsVec*EV; % [N_a,1]

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec); % [N_a,level1n], aprime-by-a

entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

%Calc the max and it's index
[Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n]

V(level1ii)=Vtempii';
Policy(level1ii)=maxindex1';

% Attempt for improved version
maxgap=maxindex1(2:end)-maxindex1(1:end-1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(ii),n_a-maxgap(ii)); % maxindex1(ii), but avoid going off top of grid when we add maxgap(ii) points
        % loweredge is 1-by-1
        aprimeindexes=loweredge+(0:1:maxgap(ii))';
        % aprime possibilities are maxgap(ii)+1-by-1
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(aprimeindexes), a_grid(curraindex), ReturnFnParamsVec); % [maxgap(ii)+1,npts]
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(aprimeindexes); % autoexpand level1iidiff(ii) in 2nd dimension
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex)=Vtempii';
        Policy(curraindex)=maxindex'+loweredge-1;
    else
        loweredge=maxindex1(ii);
        % Just use aprime(ii) for everything
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(loweredge), a_grid(curraindex), ReturnFnParamsVec); % [1,npts]
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(loweredge);
        % can skip max() over as just a single point
        V(curraindex)=entireRHS_ii';
        Policy(curraindex)=loweredge;
    end
end

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,Policy(:))',[length(n_a),N_a]);

end
