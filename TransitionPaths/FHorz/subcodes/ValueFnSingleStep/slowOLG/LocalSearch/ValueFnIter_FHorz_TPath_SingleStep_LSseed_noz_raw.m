function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, slowOLG, with a d, no z, without the grid interpolation layer. An
% UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d, [N_d,N_a,N_j], which is what
% vfoptions.localsearch needs as its reference.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw with the z dimension removed; see it.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_noz_raw: same search set, same arithmetic,
% so V is bit-identical. That raw folds (d,aprime) and takes one max; this uses the UNFOLDED Refine=1
% return matrix and maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE.
%
% j=N_j uses EV=0. R+0 is R exactly, so V still agrees bit for bit with the folded raw, which uses the
% return matrix alone at j=N_j.
% The unrestricted noz raw has no lowmemory branch, so there is none to honour here.

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,'gpuArray');

%% Iterate backwards through j, starting from j=N_j
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);

    if jj==N_j
        % terminal age has no continuation in TPath
        DiscountedEV=zeros(N_a,1,'gpuArray');
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
        Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

        DiscountedEV=DiscountFactorParamsVec*VKronNext_j; % [N_a,1], no z so no expectation
    end

    % Refine=1 keeps d and aprime in separate dimensions, (d,aprime,a), which is what the per-d max
    % below needs. EV depends on aprime, dimension 2, so it broadcasts as shiftdim(EV,-1).
    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1);

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeRefNew_j=reshape(offset1,[N_d,N_a]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);      % [1,N_a]
    V(:,jj)=reshape(Vtemp,[N_a,1]);
    dstar=reshape(dstar,[N_a,1]);

    % aprime at the chosen d. The lookup must use dstar and not the state index. Reshaped because with
    % N_d=1 aprimeRefNew_j is a row vector and a vector indexed by a vector takes the SOURCE's orientation.
    aprimestar=reshape(aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'),[N_a,1]); % [N_a,1]

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
    Policy(:,jj)=dstar+N_d*(aprimestar-1);
    aprimeReferencePolicyNew(:,:,jj)=aprimeRefNew_j;
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
