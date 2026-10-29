function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_noz_e_raw(V,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, slowOLG, with a d, no z, with e, without the grid interpolation layer.
% An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d, [N_d,N_a,N_e,N_j], which is
% what vfoptions.localsearch needs as its reference. EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw
% with e in the place of z; see it for why this exists.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_noz_e_raw: same search set, same arithmetic,
% so V is bit-identical. That raw folds (d,aprime) and takes one max; this uses the UNFOLDED level 1
% return matrix (the same arrayfun, only reshaped differently) and maxes over aprime first, then over d.
% TIE-BREAKING IS THE ONE DIFFERENCE (lowest d, then that d's lowest aprime, against linear order).
%
% j=N_j uses EV=0. R+0 is R exactly, so V still agrees bit for bit with the folded raw, which uses the
% return matrix alone at j=N_j.
%
% lowmemory=1 loops over e, exactly as the folded raw does: a seed sweep is a full sweep, and lowmemory
% is a memory option.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

Policy=zeros(N_a,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-1),2); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_LSseed_noz_e_raw')
end

%% Iterate backwards through j, starting from j=N_j
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

        EV=Vnext(:,1,jj+1); % [N_a,1]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    if vfoptions.lowmemory==0

        % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,e), which is what the per-d max
        % below needs. EV depends on aprime, dimension 2, so it broadcasts as shiftdim(EV,-1).
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1);

        % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
        [Vd,offset1]=max(entireRHS,[],2);                   % [N_d,1,N_a,N_e]
        % No window here, so the argmax is already a grid index in [1,N_a]
        aprimeRefNew_j=reshape(offset1,[N_d,N_a,N_e]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_e]),[],1);  % [1,N_a,N_e]
        V(:,:,jj)=reshape(Vtemp,[N_a,N_e]);
        dstar=reshape(dstar,[N_a,N_e]);

        % aprime at the chosen d. The lookup must use dstar and not the state index.
        aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_e-1)); % [N_a,N_e]

        % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
        Policy(:,:,jj)=dstar+N_d*(aprimestar-1);
        aprimeReferencePolicyNew(:,:,:,jj)=aprimeRefNew_j;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,1); % (d,aprime,a)
            entireRHS_e=ReturnMatrix_e+shiftdim(DiscountedEV,-1);

            [Vd,offset1]=max(entireRHS_e,[],2);                 % [N_d,1,N_a]
            aprimeRefNew_e=reshape(offset1,[N_d,N_a]);

            [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);      % [1,N_a]
            V(:,e_c,jj)=reshape(Vtemp,[N_a,1]);
            dstar=reshape(dstar,[N_a,1]);

            % aprime at the chosen d
            aprimestar=aprimeRefNew_e(dstar+N_d*(0:1:N_a-1)');  % [N_a,1]

            Policy(:,e_c,jj)=dstar+N_d*(aprimestar-1);
            aprimeReferencePolicyNew(:,:,e_c,jj)=aprimeRefNew_e;
        end

    end
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
