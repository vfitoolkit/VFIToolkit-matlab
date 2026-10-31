function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_e_raw(Vnext,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable, with z and iid e. ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw with an e
% dimension; see its header for why the reference is conditional on d, why d is searched in full, and the two maxima.
%
% THE REFERENCE IS [N_d,N_a,N_z,N_e]: one aprime index per (d,a,z,e), so per current e as well (see
% ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_e_raw for why).
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_z,N_e]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (d,a,z,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);           % [N_d,N_a,N_z,N_e]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_z,N_e])+(0:1:2*n);  % [N_d,2n+1,N_a,N_z,N_e], d fastest

%% Expected value, exactly as in the unrestricted raw: first e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z]
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only
% LEVEL 6, NOT LEVEL 3. In CreateReturnFnMatrix_Disc_DC1_e, Level 6 is 'Level 2 inputs with Level 3 outputs': it takes
% the aprime grid as given (N_aprime from its SECOND dimension) and returns [N_d,N_aprime,N_a,N_z,N_e] without folding
% d and aprime. That is what CreateReturnFnMatrix_Disc_DC1 (no e) calls Level 3, which the z version of this raw uses.
% Level 3 of the e command reads N_aprime from the FIRST dimension, so it would take N_d as the window length.
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a_grid(aprimeindexes), a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,6);

aprimez=aprimeindexes+N_a*shiftdim((0:1:N_z-1),-2); % linear index into DiscountedEV [N_a,1,N_z]; EV does not depend on e, so no e offset
entireRHS=ReturnMatrix+DiscountedEV(aprimez);

%% Two maxima, not one (as in the z version): over the window per d, which is the new reference, then over d
[Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_z,N_e]
coarseoffset=reshape(offset1,[N_d,N_a,N_z,N_e])-1;              % 0 to 2n, steps above loweredge, per d
aprimeReferencePolicyNew=loweredge+coarseoffset;

[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);              % [1,N_a,N_z,N_e]
V=reshape(V,[N_a,N_z,N_e]);
dstar=reshape(dstar,[N_a,N_z,N_e]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1));  % [N_a,N_z,N_e]

%% Policy in transition paths
% d fastest in the folded index, matching the unrestricted raw's layout
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_z,N_e]);

end
