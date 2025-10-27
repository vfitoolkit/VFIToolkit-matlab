function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z), rather than standard (a,z,j)
% V is (a,j)-by-z

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a*N_j,N_z,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=DiscountFactorParamsVec';
% DiscountFactorParamsVec=repelem(DiscountFactorParamsVec,N_a,1);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

VKronNext=[V(N_a+1:end,:); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

if vfoptions.lowmemory==0

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_z, N_j, [], a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronNext.*repelem(pi_z_J,N_a,1,1);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2);
    
    discountedEV=DiscountFactorParamsVec.*reshape(EV,[N_a,N_j,N_z]); % aprime-j-z

    entirediscountedEV=repelem(discountedEV,1,N_a,1); % (d,aprime)-by-(a,j)-by-z

    entirediscountedEV=ReturnMatrix+entirediscountedEV; %(aprime)-by-(a,j,z)

    %Calc the max and it's index
    [V,Policy]=max(entirediscountedEV,[],1);
    V=shiftdim(V,1);

elseif vfoptions.lowmemory==1

    n_z_special=ones(1,length(n_z));
    
    for z_c=1:N_z
        z_vals=z_gridvals_J(:,z_c,:); % z_gridvals_J has shape (j,prod(n_z),l_z) for fastOLG
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_z_special, N_j, [], a_grid, z_vals, ReturnFnParamsAgeMatrix);
        % fastOLG: ReturnMatrix is [aprime,a,j,z]

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=VKronNext.*repelem(pi_z_J(:,:,z_c),N_a,1,1);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);

        discountedEV_z=DiscountFactorParamsVec.*reshape(EV_z,[N_a,N_j]); % aprime-j

        entirediscountedEV_z=repelem(discountedEV_z,1,N_a); % (d,aprime)-by-(a,j)

        entirediscountedEV_z=ReturnMatrix_z+entirediscountedEV_z; %(aprime)-by-(a,j)

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entirediscountedEV_z,[],1);
        V(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;
    end

end


%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
Policy=reshape(Policy,[N_a,N_j,N_z]);
% Note that in fastOLG, we do not separate d from aprime in Policy


end
