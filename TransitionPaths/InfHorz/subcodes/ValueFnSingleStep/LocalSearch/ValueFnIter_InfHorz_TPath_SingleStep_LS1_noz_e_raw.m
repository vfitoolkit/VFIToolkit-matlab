function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_noz_e_raw(Vnext,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% Local search with a decision variable, no z, with iid e. ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw with e in the
% place of z for the return (as the noz_e raws do), and an expectation over e' that does not depend on the current e;
% see its header for why the reference is conditional on d, why d is searched in full, and the two maxima.
% THE REFERENCE IS [N_d,N_a,N_e], per current e.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

% lowmemory is not read here. The dispatcher zeroes it before calling a restricted raw, because a
% window fits in memory whatever the full sweep needs; the seed sweeps are the ones that honour it.

n=vfoptions.nlocalsearch;

aprimeReferencePolicy=reshape(aprimeReferencePolicy,[N_d,N_a,N_e]);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% The window, one per (d,a,e)
loweredge=min(max(aprimeReferencePolicy-n,1),N_a-2*n);        % [N_d,N_a,N_e]
aprimeindexes=reshape(loweredge,[N_d,1,N_a,N_e])+(0:1:2*n);   % [N_d,2n+1,N_a,N_e], d fastest

%% Expected value, exactly as in the unrestricted raw: over e'
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1]; does not depend on the current e
DiscountedEV=DiscountFactorParamsVec*EV;

%% Return matrix on the window only
% e plays the role of z: Level 3 of CreateReturnFnMatrix_Disc_DC1 is what LS1_raw uses, 'Level 2 inputs with Level 1
% outputs', [N_d,N_aprime,N_a,N_e] with N_aprime from the second dimension and d and aprime not folded
ReturnMatrix=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid, e_gridvals, ReturnFnParamsVec,3);

entireRHS=ReturnMatrix+DiscountedEV(aprimeindexes); % EV does not depend on e, so no e offset

%% Two maxima: over the window per d, which is the new reference, then over d
[Vd,offset1]=max(entireRHS,[],2);                               % [N_d,1,N_a,N_e]
aprimeReferencePolicyNew=loweredge+reshape(offset1,[N_d,N_a,N_e])-1;

[V,dstar]=max(reshape(Vd,[N_d,N_a,N_e]),[],1);                  % [1,N_a,N_e]
V=reshape(V,[N_a,N_e]);
dstar=reshape(dstar,[N_a,N_e]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_e-1)); % [N_a,N_e]

%% Policy in transition paths
% d fastest in the folded index, matching the unrestricted raw's layout
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_e]);

end
