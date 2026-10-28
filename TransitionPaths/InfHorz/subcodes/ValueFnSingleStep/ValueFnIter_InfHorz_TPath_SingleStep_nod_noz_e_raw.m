function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_nod_noz_e_raw(Vnext,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, no d, no z, with iid e.
% Vnext is [N_a,N_e]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 and 2 both loop over e.

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,'gpuArray');
Policy=zeros(N_a,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,e

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
EV=Vnext.*shiftdim(pi_e,-1); % [N_a,N_e], aprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,2); % sum over e', [N_a,1], aprime-by-1
DiscountedEV=DiscountFactorParamsVec*EV; % [N_a,1]

if vfoptions.lowmemory==0

    % e plays the role of z in the return fn matrix
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, [], a_grid, e_gridvals, ReturnFnParamsVec,0); % [N_a,N_a,N_e], aprime-by-a-by-e

    entireRHS=ReturnMatrix+DiscountedEV; % aprime-by-a-by-e (DiscountedEV autoexpands over a and e)

    %Calc the max and it's index
    [V,Policy]=max(entireRHS,[],1);
    V=reshape(V,[N_a,N_e]); % otherwise it was outputting as [1,N_a,N_e]
    Policy=reshape(Policy,[N_a,N_e]); % otherwise it was outputting as [1,N_a,N_e]

elseif vfoptions.lowmemory>=1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, [], a_grid, e_val, ReturnFnParamsVec,0); % [N_a,N_a], aprime-by-a

        entireRHS_e=ReturnMatrix_e+DiscountedEV; % aprime-by-a

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,e_c)=Vtemp;
        Policy(:,e_c)=maxindex;
    end
end

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,Policy(:))',[length(n_a),N_a,N_e]);

end
