function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_e_raw(Vnext,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with d, with z and iid e.
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e; 2 loops over z and e.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,'gpuArray');
Policy=zeros(N_a,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z,e

%%
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

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z], aprime-by-zprime
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension, [N_a,1,N_z], aprime-by-1-by-z
DiscountedEV=repelem(DiscountFactorParamsVec*EV,N_d,1,1); % [N_d*N_a,1,N_z], (d,aprime)-by-1-by-z

%%
if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec,0); % [N_d*N_a,N_a,N_z,N_e], (d,aprime)-by-a-by-z-by-e

    entireRHS=ReturnMatrix+DiscountedEV; % (d,aprime)-by-a-by-z-by-e (DiscountedEV autoexpands over a and e)

    % Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a,N_z,N_e]); % otherwise it was outputting as [1,N_a,N_z,N_e]

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals, e_val, ReturnFnParamsVec,0); % [N_d*N_a,N_a,N_z], (d,aprime)-by-a-by-z

        entireRHS_e=ReturnMatrix_e+DiscountedEV; % (d,aprime)-by-a-by-z (DiscountedEV autoexpands over a)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a,N_z]);
        Policy(:,:,e_c)=reshape(maxindex,[N_a,N_z]);
    end

elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,1,z_c); % [N_d*N_a,1]
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0); % [N_d*N_a,N_a], (d,aprime)-by-a

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z; % (d,aprime)-by-a (DiscountedEV_z autoexpands over a)

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_ze,[],1);
            V(:,z_c,e_c)=Vtemp;
            Policy(:,z_c,e_c)=maxindex;
        end
    end
end


%% Policy in transition paths
Policy2=reshape(ind2sub_vec_homemade([n_d,n_a],Policy(:))',[length(n_d)+length(n_a),N_a,N_z,N_e]);


end
