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
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a,N_z,'gpuArray');
    dstar=zeros(N_a,N_z,'gpuArray');
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,'gpuArray');
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% LOWMEMORY IS HONOURED HERE, unlike on a restricted sweep where the dispatcher zeroes it. This is
%% the full sweep, and it holds the LARGEST array in the whole local search feature: the level 1
%% matrix is [N_d,N_a,N_a,N_z], which at N_d=51, N_a=1001, N_z=5 is 255 million doubles, about 2 GB.
%% Looping over z brings that to one [N_d,N_a,N_a] slice at a time, about 409 MB. Without this branch
%% the promise that lowmemory applies to the full sweep would be broken in exactly the corner where
%% the option gets set -- with a d, no interpolation layer, on a big grid.
%% The two branches are the same arithmetic and differ only in how much of it is resident, so they
%% agree exactly; the test bank holds them to that.
if vfoptions.lowmemory==0

    %% Expected value
    EV=Vnext.*shiftdim(pi_z',-1);
    EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % [N_a,1,N_z]

    % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,z), which is what the per-d max
    % below needs. The folded raw uses level 0 and repelem(EV,N_d,1,1); unfolded, EV broadcasts.
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV,-1);

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_z]

    % The new reference: the aprime argmax per (d,a,z), over the WHOLE grid. There is no window here,
    % so the argmax is already a grid index in [1,N_a] and needs no loweredge offset -- unlike the
    % restricted raws, where coarseoffset is 0-based and the index is loweredge+coarseoffset.
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_z]);

    % Then the max over d
    [V,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);      % [1,N_a,N_z]
    V=reshape(V,[N_a,N_z]);
    dstar=reshape(dstar,[N_a,N_z]);

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);

        EV_z=Vnext.*pi_z(z_c,:);
        EV_z(isnan(EV_z))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_z=sum(EV_z,2); % [N_a,1]

        ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
        % [N_d,N_a,N_a] against shiftdim(EV_z,-1)=[1,N_a,1]: EV depends on aprime, which is dimension
        % 2, and not on a, which is dimension 3. Same broadcast as the lowmemory==0 branch above.
        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*shiftdim(EV_z,-1);

        [Vd_z,offset1_z]=max(entireRHS_z,[],2);         % [N_d,1,N_a]
        aprimeReferencePolicyNew(:,:,z_c)=reshape(offset1_z,[N_d,N_a]);

        [V_z,dstar_z]=max(reshape(Vd_z,[N_d,N_a]),[],1); % [1,N_a]
        V(:,z_c)=V_z';
        dstar(:,z_c)=dstar_z';
    end

end

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1);     % [N_a,N_z], into an [N_d,N_a,N_z] array
aprimestar=aprimeReferencePolicyNew(lin);

%% Policy, in the same form and with the same interpretation as the folded raw's Policy2
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_z]);

end
