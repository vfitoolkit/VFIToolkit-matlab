function [V,Policy2]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e_raw(Vnext,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with d, no z, with iid e.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e. (2 loops over z and e, so there is no lowmemory=2 without z; the callers refuse it.)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,'gpuArray');
Policy=zeros(N_a,N_e,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,e

%%
if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the expectation term (except beta), computed once and used by every lowmemory branch
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1], aprime-by-1
DiscountedEV=repelem(DiscountFactorParamsVec*EV,N_d,1); % [N_d*N_a,1], (d,aprime)-by-1

%%
if vfoptions.lowmemory==0

    % e plays the role of z in the return fn matrix
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals, ReturnFnParamsVec,0); % [N_d*N_a,N_a,N_e], (d,aprime)-by-a-by-e

    entireRHS=ReturnMatrix+DiscountedEV; % (d,aprime)-by-a-by-e (DiscountedEV autoexpands over a and e)

    % Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a,N_e]); % otherwise it was outputting as [1,N_a,N_e]

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0); % [N_d*N_a,N_a], (d,aprime)-by-a

        entireRHS_e=ReturnMatrix_e+DiscountedEV; % (d,aprime)-by-a

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,e_c)=Vtemp;
        Policy(:,e_c)=maxindex;
    end
end


%% Policy in transition paths
Policy2=reshape(ind2sub_vec_homemade([n_d,n_a],Policy(:))',[length(n_d)+length(n_a),N_a,N_e]);


end
