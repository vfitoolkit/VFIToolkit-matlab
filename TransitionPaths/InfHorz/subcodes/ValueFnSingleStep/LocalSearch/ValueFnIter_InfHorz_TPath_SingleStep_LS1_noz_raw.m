function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_noz_raw(Vnext,n_d,n_a, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable, no z, no e. ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw without the z
% dimension; see its header for why the reference is conditional on d, why d is searched in full, and the two maxima.
% The reference is [N_d,N_a].
% Vnext is [N_a,1].

N_d=prod(n_d);
N_a=prod(n_a);

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (d,a)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);     % [N_d,N_a]
aprimeindexes=reshape(loweredge,[N_d,1,N_a])+(0:1:2*n);    % [N_d,2n+1,N_a], d fastest

% No shocks, so the expectation is just next period value fn
DiscountedEV=DiscountFactorParamsVec*Vnext; % [N_a,1]

%% Return matrix on the window only
% Level 3 of CreateReturnFnMatrix_Disc_DC1_noz takes the aprime grid as given, [N_d,N_aprime,(N_a)] with N_aprime from
% its SECOND dimension, and returns [N_d,N_aprime,N_a] without folding d and aprime
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid, ReturnFnParamsVec,3);

entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes);

%% Two maxima: over the window per d, which is the new reference, then over d
[Vd,offset1]=max(entireRHS,[],2);                      % [N_d,1,N_a]
aprimeReferencePolicyNew=loweredge+reshape(offset1,[N_d,N_a])-1;

[V,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);             % [1,N_a]
V=reshape(V,[N_a,1]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)); % [1,N_a]

%% Policy in transition paths
% d fastest in the folded index, matching the unrestricted raw's layout
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a]);

end
