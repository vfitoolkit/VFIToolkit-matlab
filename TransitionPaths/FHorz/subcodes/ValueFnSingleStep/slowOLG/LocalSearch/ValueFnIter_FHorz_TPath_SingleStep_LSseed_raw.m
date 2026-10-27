function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, slowOLG, with a d and without the grid interpolation layer. An
% UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d, [N_d,N_a,N_z,N_j], which is what
% vfoptions.localsearch needs as its reference. Reached only through vfoptions.localsearchseed, which
% Step1 sets on the standard sweeps of a localsearch solve.
% Each age is EXACTLY ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw; see it for why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_raw: same search set, same arithmetic, so V
% is bit-identical. That raw folds (d,aprime) and takes one max; this uses the UNFOLDED level 1 return
% matrix and maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE (lowest d, then
% that d's lowest aprime, against linear order). The folded raw is not changed to do it this way: it is
% used by everything, and a tie-breaking change there would move policies across every test bank.
%
% j=N_j uses EV=0. R+0 is R exactly, so V still agrees bit for bit with the folded raw, which uses the
% return matrix alone at j=N_j.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% No lowmemory branch. On the slowOLG layout TransitionPath_Case1_FHorz refuses lowmemory>0 outright,
% and this raw is reachable only from a TransitionPath_Case1_FHorz solve.
if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw')
end

Policy=zeros(N_a,N_z,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_j,'gpuArray');

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountedEV=zeros(N_a,1,N_z,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % [N_a,1,N_z]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,z), which is what the per-d max
    % below needs. EV depends on aprime, dimension 2, so it broadcasts as shiftdim(EV,-1).
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1);

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_z]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeRefNew_j=reshape(offset1,[N_d,N_a,N_z]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);  % [1,N_a,N_z]
    V(:,:,jj)=reshape(Vtemp,[N_a,N_z]);
    dstar=reshape(dstar,[N_a,N_z]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)); % [N_a,N_z]

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
    Policy(:,:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
