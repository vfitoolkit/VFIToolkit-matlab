function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, fastOLG (all ages at once), with z and e, with a d and without the
% grid interpolation layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d,
% [N_d,N_a,N_j,N_z,N_e], which is what vfoptions.localsearch needs as its reference.
% This is EXACTLY slowOLG/LocalSearch/ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw, all ages at once
% and with e; see it for why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e_raw: same search set, same
% arithmetic, so V is bit-identical. That raw folds (d,aprime) and takes one max; this reshapes the
% same folded return matrix (the fold is d fastest, so the reshape to [N_d,N_aprime,...] is exact) and
% maxes over aprime first, then over d. TIE-BREAKING IS THE ONE DIFFERENCE (lowest d, then that d's
% lowest aprime, against linear order).
%
% A seed sweep is a FULL sweep, so lowmemory is honoured here, with the same loops over e (=1) and over
% z and e (=2) as the unrestricted raw.
% j=N_j: EV is zero there via the zero slice appended to pi_z_J, as in the unrestricted raw. ONLY THE
% EVpre=0 PATH, which is all that raw has.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray')); % append a j=N_j slot of zeros: there is no continuation value in the final period (input pi_z_J has N_j-1 slices, the transitions from periods 1..N_j-1)
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-3);
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,N_j,1,N_e,length(n_e)]);

Policy=zeros(1,N_a,N_j,N_z,N_e,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_j,N_z,N_e,'gpuArray');

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactor_J=prod(CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j),2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(N_a+1:end,:,:),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations
EVpre=reshape(EVpre,[N_a,1,N_j,N_z]);
EV=EVpre.*shiftdim(pi_z_J,-2);
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=reshape(sum(EV,4),[N_a,1,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a

DiscountedEV=repelem(reshape(DiscountFactor_J,[1,1,N_j]).*EV,N_d,1,1,1); % [N_d*N_aprime,1,N_j,N_z]
% Unfolded, d fastest, as the per-d max needs it. The repelem copies each value across d, so this is
% the same numbers in the same places as the folded raw adds.
DiscountedEV=reshape(DiscountedEV,[N_d,N_a,1,N_j,N_z]);

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [d*aprime,a,j,z,e], d fastest

    entireRHS=reshape(ReturnMatrix,[N_d,N_a,N_a,N_j,N_z,N_e])+DiscountedEV; % (d,aprime,a,j,z,e)

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,aprimeRefNew]=max(entireRHS,[],2);           % [N_d,1,N_a,N_j,N_z,N_e]
    % No window here, so the argmax is already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(aprimeRefNew,[N_d,N_a,N_j,N_z,N_e]);

    [Vtemp,dstar]=max(Vd,[],1);                      % [1,1,N_a,N_j,N_z,N_e]
    V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
    dstar=reshape(dstar,[1,N_a,N_j,N_z,N_e]);

    % aprime at the chosen d. The lookup must use dstar and not the state index.
    aprimestar=aprimeReferencePolicyNew(dstar+N_d*reshape(gpuArray(0:1:N_a*N_j*N_z*N_e-1),[1,N_a,N_j,N_z,N_e]));

    % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
    Policy=dstar+N_d*(aprimestar-1);

elseif vfoptions.lowmemory==1

    special_n_e=ones(1,length(n_e));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_gridvals_J, e_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [d*aprime,a,j,z], d fastest

        entireRHS_e=reshape(ReturnMatrix_e,[N_d,N_a,N_a,N_j,N_z])+DiscountedEV; % (d,aprime,a,j,z)

        [Vd,aprimeRefNew]=max(entireRHS_e,[],2);     % [N_d,1,N_a,N_j,N_z]
        aprimeRefNew=reshape(aprimeRefNew,[N_d,N_a,N_j,N_z]);
        aprimeReferencePolicyNew(:,:,:,:,e_c)=aprimeRefNew;

        [Vtemp,dstar]=max(Vd,[],1);                  % [1,1,N_a,N_j,N_z]
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        dstar=reshape(dstar,[N_a,N_j,N_z]);

        aprimestar=aprimeRefNew(dstar+N_d*reshape(gpuArray(0:1:N_a*N_j*N_z-1),[N_a,N_j,N_z])); % [N_a,N_j,N_z]
        Policy(1,:,:,:,e_c)=shiftdim(dstar+N_d*(aprimestar-1),-1);
    end

elseif vfoptions.lowmemory==2

    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_vals=z_gridvals_J(1,1,1,:,z_c,:); % z_gridvals_J has shape (1,1,1,j,prod(n_z),l_z) for fastOLG with d
        DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
        for e_c=1:N_e
            e_vals=e_gridvals_J(1,1,1,:,1,e_c,:); % e_gridvals_J has shape (1,1,1,j,1,prod(n_e),l_e) for fastOLG with d

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, N_j, d_gridvals, a_grid, a_grid, z_vals, e_vals, ReturnFnParamsAgeMatrix);
            % fastOLG: ReturnMatrix is [d*aprime,a,j], d fastest

            entireRHS_ze=reshape(ReturnMatrix_ze,[N_d,N_a,N_a,N_j])+DiscountedEV_z; % (d,aprime,a,j)

            [Vd,aprimeRefNew]=max(entireRHS_ze,[],2); % [N_d,1,N_a,N_j]
            aprimeRefNew=reshape(aprimeRefNew,[N_d,N_a,N_j]);
            aprimeReferencePolicyNew(:,:,:,z_c,e_c)=aprimeRefNew;

            [Vtemp,dstar]=max(Vd,[],1);              % [1,1,N_a,N_j]
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            dstar=reshape(dstar,[N_a,N_j]);

            aprimestar=aprimeRefNew(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_j-1)); % [N_a,N_j]
            Policy(1,:,:,z_c,e_c)=shiftdim(dstar+N_d*(aprimestar-1),-1);
        end
    end

else
    error('vfoptions.lowmemory>2 not supported for ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_e_raw (nor by the unrestricted fastOLG_e raw)')
end

end
