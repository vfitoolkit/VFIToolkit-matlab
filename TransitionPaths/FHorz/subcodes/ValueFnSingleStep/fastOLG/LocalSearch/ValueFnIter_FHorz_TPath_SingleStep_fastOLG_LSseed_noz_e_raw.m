function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, fastOLG (all ages at once), with a d, no z, with e, and without the
% grid interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_e], (a,j,e) being the order of Policy, which is what vfoptions.localsearch needs as its
% reference. Reached only through vfoptions.localsearchseed, which Step1 sets on the standard sweeps of
% a localsearch solve. Done exactly as slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw;
% see it (and the InfHorz original) for why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e_raw: same return matrix, same
% sum, same search set, so V is bit-identical. That raw takes one max over the folded (d,aprime); this
% reshapes the very same folded RHS to [N_d,N_aprime,...] (the fold is d fastest, so this is exact) and
% maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE (lowest d, then that d's
% lowest aprime, against linear order). The folded raw is not changed to do it this way: it is used by
% everything, and a tie-breaking change there would move policies across every test bank.
%
% Policy comes out exactly as the unrestricted raw's, branch by branch: [1,N_a,N_j,N_e] at lowmemory=0,
% [1,N_a*N_j,N_e] at lowmemory=1 (the same linear order). Every lowmemory branch of the unrestricted raw
% is honoured, because a seed sweep is a full sweep.
% The terminal age needs nothing: as in the unrestricted raw, EV is zero at j=N_j.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

e_gridvals_J=shiftdim(e_gridvals_J,-3); % [1,1,1,N_j,N_e,l_e]

%% First, create the big 'next period (of transition path) expected value fn.
% Exactly as the unrestricted raw
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EV=[sum(V(N_a+1:end,:).*pi_e_J(N_a+1:end,:),2); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_e_J to create expectations

discountedEV=repelem(reshape(DiscountFactor_J,[1,1,N_j]).*reshape(EV,[N_a,1,N_j]),N_d,1,1); % [d & aprime, 1, j]

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_e, N_j, d_gridvals, a_grid, a_grid, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d*aprime,a,j,e], d fastest

    entirediscountedEV=ReturnMatrix+discountedEV; %(d,aprime)-by-(a,j,e)

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(reshape(entirediscountedEV,[N_d,N_a,N_a,N_j,N_e]),[],2); % [N_d,1,N_a,N_j,N_e]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_j,N_e]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_e]),[],1); % [1,N_a,N_j,N_e]
    V=reshape(Vtemp,[N_a*N_j,N_e]);
    dstar=reshape(dstar,[N_a,N_j,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim(0:1:N_e-1,-1)); % [N_a,N_j,N_e]

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy. The unrestricted
    % raw squeezes its [1,N_a,N_j,N_e]; squeeze of this reshape gives the same array.
    Policy=squeeze(reshape(dstar+N_d*(aprimestar-1),[1,N_a,N_j,N_e]));

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));

    Policy=zeros(N_a*N_j,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,e_c,:); % e_gridvals_J has shape (j,prod(n_e),l_e) for fastOLG with no z

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_e, N_j, d_gridvals, a_grid, a_grid, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d*aprime,a,j], d fastest

        entirediscountedEV_e=ReturnMatrix_e+discountedEV; %(d,aprime)-by-(a,j)

        % Two maxima, as above
        [Vd,offset1]=max(reshape(entirediscountedEV_e,[N_d,N_a,N_a,N_j]),[],2); % [N_d,1,N_a,N_j]
        aprimeRefNew_e=reshape(offset1,[N_d,N_a,N_j]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1); % [1,N_a,N_j]
        V(:,e_c)=reshape(Vtemp,[N_a*N_j,1]);
        dstar=reshape(dstar,[N_a,N_j]);

        % aprime at the chosen d. The lookup must use dstar and not the state index.
        aprimestar=aprimeRefNew_e(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)); % [N_a,N_j]

        Policy(:,e_c)=reshape(dstar+N_d*(aprimestar-1),[N_a*N_j,1]);
        aprimeReferencePolicyNew(:,:,:,e_c)=aprimeRefNew_e;
    end
end

% Output shape for policy, the same as the unrestricted raw
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
