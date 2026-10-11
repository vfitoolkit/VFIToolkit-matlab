function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw(Vnext,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, with a d and without the grid interpolation layer. An UNRESTRICTED sweep that also
% returns the aprime argmax FOR EACH d, which is what vfoptions.localsearch needs as its reference.
% Reached only through vfoptions.localsearchseed, which Step1 sets on the standard sweeps of a
% localsearch solve. With the interpolation layer the standard GI raw already computes a per-d coarse
% argmax (its midpoint) and returns it as a third output, so no seed raw is needed there.
%
% WHY THIS EXISTS. A standard sweep returns one (dstar,aprimestar) per state, so it fixes the
% reference only for the d it chose and says nothing about the other N_d-1 values. The next
% restricted sweep searches a window per d, so it needs a centre per d. That is the whole of the gap
% that kept localsearch out of a general eqm path with a d.
%
% HOW IT DIFFERS FROM ValueFnIter_InfHorz_TPath_SingleStep_raw, which it has to agree with. That raw
% folds (d,aprime) into one dimension and takes a single max, so no per-d argmax exists anywhere in
% it. This one uses the UNFOLDED level 1 return matrix, (d,aprime,a,z), and takes the max over aprime
% first -- giving a value and an argmax per d -- and only then the max over d. Same search set, same
% arithmetic, so V is bit-identical.
% TIE-BREAKING IS THE ONE DIFFERENCE. A folded max breaks a tie in linear order, which runs d fastest
% and so picks the lowest aprime and then the lowest d within it; this picks the lowest d and then
% that d's lowest aprime. Two names for one value. The test bank holds V to an identity against the
% folded raw and reports the policy difference count, which should be ties only.
% That is also why the folded raw is NOT changed to do it this way: it is used by everything, and a
% tie-breaking change there would move policies across every test bank for no gain.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.lowmemory>0
    error('vfoptions.lowmemory is not used with vfoptions.localsearch=1 (the seed sweep of a localsearch solve holds the whole unrestricted return matrix, as the standard sweep it replaces does)')
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value
EV=Vnext.*shiftdim(pi_z',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]

%% Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,z), which is what the per-d max
%% below needs. The folded raw uses level 0 and repelem(EV,N_d,1,1); unfolded, EV broadcasts.
ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,1);
entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

%% Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
[Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_z]

%% The new reference: the aprime argmax per (d,a,z), over the WHOLE grid
% There is no window here, so the argmax is already a grid index in [1,N_a] and needs no loweredge
% offset -- unlike the restricted raws, where coarseoffset is 0-based and the index is
% loweredge+coarseoffset.
aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_z]);

%% Then the max over d
[V,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);      % [1,N_a,N_z]
V=reshape(V,[N_a,N_z]);
dstar=reshape(dstar,[N_a,N_z]);

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1);     % [N_a,N_z], into an [N_d,N_a,N_z] array
aprimestar=aprimeReferencePolicyNew(lin);

%% Policy, in the same form and with the same interpretation as the folded raw's Policy2
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_z]);

end
