function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_raw(V,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, fastOLG (all ages at once), with z, with a d and without the grid
% interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_z], which is what vfoptions.localsearch needs as its reference. Reached only through
% vfoptions.localsearchseed, which Step1 sets on the standard sweeps of a localsearch solve.
% This is EXACTLY slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw with every age in
% one kernel; see it for why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_fastOLG_raw: same search set, same creator,
% same arithmetic, so V is bit-identical. That raw folds (d,aprime) and takes one max; this reshapes
% the SAME folded return matrix to [N_d,N_aprime,...] (the fold is d fastest, so the reshape is exact)
% and maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE (lowest d, then that d's
% lowest aprime, against linear order). The folded raw is not changed to do it this way: it is used by
% everything, and a tie-breaking change there would move policies across every test bank.
%
% A seed sweep is a FULL sweep, so it honours the same lowmemory branches as the folded raw (lowmemory=1
% loops over z).
% j=N_j: EV is zero there via the zero slice appended to pi_z_J, as in the folded raw.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% ONLY THE vfoptions.EVpre==0 PATH. Local search is reachable only from a transition path, where
% EVpre=0 (EVpre=1 is the Matched Expectations Path, which never sets localsearch).
if vfoptions.EVpre~=0
    error('vfoptions.localsearch requires vfoptions.EVpre=0 (ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_raw)')
end
if vfoptions.lowmemory>1
    error('vfoptions.lowmemory>1 not supported for ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_raw')
end

z_gridvals_J=shiftdim(z_gridvals_J,-3); % [1,1,1,N_j,N_z,l_z]

%% First, create the big 'next period (of transition path) expected value fn.
DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
EVpre=zeros(N_a,1,N_j,N_z);
EVpre(:,1,1:N_j-1,:)=reshape(V(N_a+1:end,:),[N_a,1,N_j-1,N_z]); % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z)

% EV depends on aprime, which is dimension 2 of the unfolded return matrix, so it broadcasts as
% shiftdim(DiscountedEV,-1). Same products as the folded raw's repelem(DiscountedEV,N_d,1,1,1).
DiscountedEV=reshape(DiscountFactor_J,[1,1,N_j]).*EV; % [N_a,1,N_j,N_z]

if vfoptions.lowmemory==0

    % The folded raw's creator, unfolded by reshape: [d,aprime,a,j,z]
    ReturnMatrix=reshape(CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_z, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix),[N_d,N_a,N_a,N_j,N_z]);

    entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1); % [d,aprime,a,j,z]

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                           % [N_d,1,N_a,N_j,N_z]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_j,N_z]);

    [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j,N_z]),[],1);      % [1,N_a,N_j,N_z]
    V=reshape(Vtemp,[N_a*N_j,N_z]);
    dstar=reshape(dstar,[N_a,N_j,N_z]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeReferencePolicyNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)+N_d*N_a*N_j*shiftdim((0:1:N_z-1),-1)); % [N_a,N_j,N_z]

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
    Policy=dstar+N_d*(aprimestar-1); % [N_a,N_j,N_z]

elseif vfoptions.lowmemory==1

    Policy=zeros(N_a,N_j,N_z,'gpuArray');
    aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,'gpuArray');

    special_n_z=ones(1,length(n_z));

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,N_j,N_z,l_z) for fastOLG
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        % The folded raw's creator, unfolded by reshape: [d,aprime,a,j]
        ReturnMatrix_z=reshape(CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, special_n_z, N_j, d_gridvals, a_grid, a_grid, z_vals, ReturnFnParamsAgeMatrix),[N_d,N_a,N_a,N_j]);

        entireRHS_z=ReturnMatrix_z+shiftdim(DiscountedEV_z,-1); % [d,aprime,a,j]

        [Vd,offset1]=max(entireRHS_z,[],2);                         % [N_d,1,N_a,N_j]
        aprimeRefNew_z=reshape(offset1,[N_d,N_a,N_j]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_j]),[],1);          % [1,N_a,N_j]
        V(:,z_c)=reshape(Vtemp,[N_a*N_j,1]);
        dstar=reshape(dstar,[N_a,N_j]);

        % aprime at the chosen d. The lookup must use dstar and not the state index.
        aprimestar=aprimeRefNew_z(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)); % [N_a,N_j]

        Policy(:,:,z_c)=dstar+N_d*(aprimestar-1);
        aprimeReferencePolicyNew(:,:,:,z_c)=aprimeRefNew_z;
    end
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1); % [1,N_a,N_j,N_z]

end
