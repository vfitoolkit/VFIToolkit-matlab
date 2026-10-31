function [V, Policy, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_e_raw(Vnext,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% THE SEED SWEEP, with a d, with z and iid e, without the grid interpolation layer. An UNRESTRICTED sweep that also
% returns the aprime argmax FOR EACH d, [N_d,N_a,N_z,N_e], which is what vfoptions.localsearch needs as its reference.
% ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw with an e dimension; see its header for why this exists, why it uses
% the UNFOLDED return matrix, and the one difference from the folded raw (tie-breaking).
% It has to agree with ValueFnIter_InfHorz_TPath_SingleStep_e_raw, including in every lowmemory branch.
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e; 2 loops over z and e (as the e raw).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,'gpuArray');
dstar=zeros(N_a,N_z,N_e,'gpuArray');
aprimeReferencePolicyNew=zeros(N_d,N_a,N_z,N_e,'gpuArray');

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%% Expected value, exactly as in the e raw: first e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z]
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % [N_a,1,N_z]
% Unfolded, (d,aprime,a,z): EV depends on aprime (dimension 2) and z (dimension 4), so it broadcasts over d, a and e
DiscountedEV=DiscountFactorParamsVec*shiftdim(EV,-1); % [1,N_a,1,N_z]

%% LOWMEMORY IS HONOURED HERE, as in the z seed raw: this is the full sweep, and the level 1 matrix
%% [N_d,N_a,N_a,N_z,N_e] is the largest array in the local search feature.
if vfoptions.lowmemory==0

    % Level 1 keeps d and aprime in separate dimensions, (d,aprime,a,z,e)
    ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,1);
    entireRHS=ReturnMatrix+DiscountedEV;

    % Two maxima. The first is over aprime FOR EACH d and is the whole point of this raw.
    [Vd,offset1]=max(entireRHS,[],2);                    % [N_d,1,N_a,N_z,N_e]

    % The new reference: the aprime argmax per (d,a,z,e), over the WHOLE grid, so already a grid index in [1,N_a]
    aprimeReferencePolicyNew=reshape(offset1,[N_d,N_a,N_z,N_e]);

    % Then the max over d
    [V,dstar]=max(reshape(Vd,[N_d,N_a,N_z,N_e]),[],1);   % [1,N_a,N_z,N_e]
    V=reshape(V,[N_a,N_z,N_e]);
    dstar=reshape(dstar,[N_a,N_z,N_e]);

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals, e_val, ReturnFnParamsVec,1); % [N_d,N_a,N_a,N_z]

        entireRHS_e=ReturnMatrix_e+DiscountedEV;

        [Vd_e,offset1_e]=max(entireRHS_e,[],2);          % [N_d,1,N_a,N_z]
        aprimeReferencePolicyNew(:,:,:,e_c)=reshape(offset1_e,[N_d,N_a,N_z]);

        [V_e,dstar_e]=max(reshape(Vd_e,[N_d,N_a,N_z]),[],1); % [1,N_a,N_z]
        V(:,:,e_c)=reshape(V_e,[N_a,N_z]);
        dstar(:,:,e_c)=reshape(dstar_e,[N_a,N_z]);
    end

elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(1,:,1,z_c); % [1,N_a], depends on aprime (dimension 2) only
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,1); % [N_d,N_a,N_a]

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; % autoexpands over d and a

            [Vd_ze,offset1_ze]=max(entireRHS_ze,[],2);   % [N_d,1,N_a]
            aprimeReferencePolicyNew(:,:,z_c,e_c)=reshape(offset1_ze,[N_d,N_a]);

            [V_ze,dstar_ze]=max(reshape(Vd_ze,[N_d,N_a]),[],1); % [1,N_a]
            V(:,z_c,e_c)=V_ze';
            dstar(:,z_c,e_c)=dstar_ze';
        end
    end
end

% aprime at the chosen d. The lookup must use dstar and not the state index, which is the one thing
% to get wrong here.
lin=dstar+N_d*(0:1:N_a-1)'+N_d*N_a*(0:1:N_z-1)+N_d*N_a*N_z*shiftdim((0:1:N_e-1),-1); % [N_a,N_z,N_e], into an [N_d,N_a,N_z,N_e] array
aprimestar=aprimeReferencePolicyNew(lin);

%% Policy, in the same form and with the same interpretation as the folded raw's Policy2
Policy=reshape(ind2sub_vec_homemade([n_d,n_a],reshape(dstar+N_d*(aprimestar-1),[],1))',[length(n_d)+length(n_a),N_a,N_z,N_e]);

end
