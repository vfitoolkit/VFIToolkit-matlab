function [V,Policy,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_e_raw(V,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, finite horizon, slowOLG, with z and e, with a d and without the grid interpolation
% layer. An UNRESTRICTED sweep that also returns the aprime argmax FOR EACH d, [N_d,N_a,N_z,N_e,N_j],
% which is what vfoptions.localsearch needs as its reference.
% EXACTLY ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw (see it for why this exists) with e added.
%
% It has to agree with ValueFnIter_FHorz_TPath_SingleStep_e_raw: same search set, same arithmetic, same
% EV (expectation over e UP FRONT, Vnext), so V is bit-identical. That raw folds (d,aprime) and takes
% one max; this uses the UNFOLDED return matrix (Refine=1) and maxes over aprime first, then over d.
% TIE-BREAKING IS THE ONE DIFFERENCE. j=N_j uses EV=0, R+0 is R exactly.
%
% Every lowmemory branch of the e raw is honoured (a seed is a full sweep, lowmemory is a memory option).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end
if vfoptions.lowmemory>=3
    error('vfoptions.lowmemory=K not supported for ValueFnIter_FHorz_TPath_SingleStep_LSseed_e_raw')
end

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_e,N_j,'gpuArray');

Vnext=sum(V.*shiftdim(pi_e_J,-2),3); % Take expectations over e: Vnext(...,jj+1) is read for current age jj, so weight V at age jj+1 by pi_e_J(:,jj+1), the distribution of the e realized in period jj+1; first column is never read

%% Iterate backwards through j, starting from j=N_j
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

        VKronNext_j=Vnext(:,:,1,jj+1);

        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % [N_a,1,N_z]
        DiscountedEV=DiscountFactorParamsVec*EV;
    end

    if vfoptions.lowmemory==0

        % Refine=1 keeps d and aprime in separate dimensions, (d,aprime,a,z,e). EV depends on (aprime,z),
        % so it broadcasts as shiftdim(EV,-1).
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1);

        % Two maxima. The first is over aprime FOR EACH d.
        [Vd,offset1]=max(entireRHS,[],2);                        % [N_d,1,N_a,N_z,N_e]
        % No window here, so the argmax is already a grid index in [1,N_a]
        aprimeRefNew_j=reshape(offset1,[N_d,N_a,N_z,N_e]);

        [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);   % [1,N_a,N_z,N_e]
        V(:,:,:,jj)=reshape(Vtemp,[N_a,N_z,N_e]);
        dstar=reshape(dstar,[N_a,N_z,N_e]);

        % aprime at the chosen d. The lookup must use dstar and not the state index.
        aprimestar=aprimeRefNew_j(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1)); % [N_a,N_z,N_e]

        % The folded (d,aprime) index, d fastest, the same form as the folded raw's Policy
        Policy(:,:,:,jj)=dstar+N_d*(aprimestar-1);
        aprimeReferencePolicyNew(:,:,:,:,jj)=aprimeRefNew_j;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
            % (d,aprime,a,z)
            entireRHS_e=ReturnMatrix_e+shiftdim(DiscountedEV,-1);

            [Vd,offset1]=max(entireRHS_e,[],2);                      % [N_d,1,N_a,N_z]
            aprimeRefNew_e=reshape(offset1,[N_d,N_a,N_z]);

            [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a,N_z]),[],1);       % [1,N_a,N_z]
            V(:,:,e_c,jj)=reshape(Vtemp,[N_a,N_z]);
            dstar=reshape(dstar,[N_a,N_z]);

            aprimestar=aprimeRefNew_e(dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)); % [N_a,N_z]

            Policy(:,:,e_c,jj)=dstar+N_d*(aprimestar-1);
            aprimeReferencePolicyNew(:,:,:,e_c,jj)=aprimeRefNew_e;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1);
                % (d,aprime,a)
                entireRHS_ze=ReturnMatrix_ze+shiftdim(DiscountedEV_z,-1);

                [Vd,offset1]=max(entireRHS_ze,[],2);                 % [N_d,1,N_a]
                aprimeRefNew_ze=reshape(offset1,[N_d,N_a]);

                [Vtemp,dstar]=max(reshape(Vd,[N_d,N_a]),[],1);       % [1,N_a]
                V(:,z_c,e_c,jj)=Vtemp';
                dstar=dstar';

                aprimestar=aprimeRefNew_ze(dstar+N_d*(0:1:N_a-1)');  % [N_a,1]

                Policy(:,z_c,e_c,jj)=dstar+N_d*(aprimestar-1);
                aprimeReferencePolicyNew(:,:,z_c,e_c,jj)=aprimeRefNew_ze;
            end
        end

    end
end

%% Output shape for policy, the same as the folded raw
Policy=shiftdim(Policy,-1);

end
