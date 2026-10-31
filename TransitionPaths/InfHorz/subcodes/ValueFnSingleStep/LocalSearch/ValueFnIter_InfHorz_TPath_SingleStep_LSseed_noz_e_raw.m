function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_noz_e_raw(Vnext,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, with a d, no z, with iid e, without the grid interpolation layer. An UNRESTRICTED sweep that also
% returns the aprime argmax FOR EACH d, [N_d,N_a,N_e], which is what vfoptions.localsearch needs as its reference.
% ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw with e in the place of z for the return; see its header for why
% this exists, why it uses the UNFOLDED return matrix, and the one difference from the folded raw (tie-breaking).
% It has to agree with ValueFnIter_InfHorz_TPath_SingleStep_noz_e_raw, including in every lowmemory branch.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e (as the noz_e raw; there is no lowmemory=2 without z).

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,'gpuArray');
dstar=zeros(N_a,N_e,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_e,'gpuArray');

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value, exactly as in the noz_e raw: over e'
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1]
% Unfolded, (d,aprime,a,e): EV depends on aprime (dimension 2) only, so it broadcasts over d, a and e
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,N_a]

if vfoptions.lowmemory==0

    % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,e); e plays the role of z in the return fn matrix
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountedEV;

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                    % [N_d,1,N_a,N_e]

    % The new reference: the aprime argmax per (d,a,e), over the WHOLE grid, so already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_e]);

    % Then the max over d
    [V,dstar]=max(reshape(Vd,[N_d,N_a,N_e]),[],1);       % [1,N_a,N_e]
    V=reshape(V,[N_a,N_e]);
    dstar=reshape(dstar,[N_a,N_e]);

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1); % [N_d,N_a,N_a]

        entireRHS_e=ReturnMatrix_e+DiscountedEV;

        [Vd_e,offset1_e]=max(entireRHS_e,[],2);          % [N_d,1,N_a]
        aprimeReferencePolicyNew(:,:,e_c)=reshape(offset1_e,[N_d,N_a]);

        [V_e,dstar_e]=max(reshape(Vd_e,[N_d,N_a]),[],1); % [1,N_a]
        V(:,e_c)=V_e';
        dstar(:,e_c)=dstar_e';
    end
end

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_e-1); % [N_a,N_e], into an [N_d,N_a,N_e] array
aprimestar=aprimeReferencePolicyNew(lin);

%% Policy, in the same form and with the same interpretation as the folded raw's Policy2
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_e]);

end
