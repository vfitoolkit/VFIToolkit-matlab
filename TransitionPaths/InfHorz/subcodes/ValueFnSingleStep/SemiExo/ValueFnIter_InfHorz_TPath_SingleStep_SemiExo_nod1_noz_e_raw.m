function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_nod1_noz_e_raw(Vnext,n_d2,n_a,n_semiz,n_e, d2_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with semi-exogenous state semiz, no d1, with d2 (d2 determines the transition of semiz), no z, with iid e.
% Vnext is [N_a,N_semiz,N_e] (next period value fn)
% pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e.
% Policy is returned as kron rows [d2; aprime], [2,N_a,N_semiz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

% Preallocate
V_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz,N_e]).*reshape(pi_e,[1,1,N_e]); % [N_a,N_semiz,N_e], aprime-by-semizprime-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_semiz]
EV=EV.*permute(pi_semiz,[4,2,1,3]); % [N_a,N_semizprime,N_semiz,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_semiz,N_d2]); % aprime-by-semiz-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, n_e, d2_val, a_grid, semiz_gridvals, e_gridvals, ReturnFnParamsVec,0); % [N_a,N_a,N_semiz,N_e], aprime-by-a-by-semiz-by-e

        entireRHS=ReturnMatrix_d2+reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_semiz]); % autoexpands over a and e

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V_ford2(:,:,:,d2_c)=reshape(Vtemp,[N_a,N_semiz,N_e]);
        Policy_ford2(:,:,:,d2_c)=reshape(maxindex,[N_a,N_semiz,N_e]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);
        DiscountedEV_d2=reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_semiz]);

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_semiz, special_n_e, d2_val, a_grid, semiz_gridvals, e_val, ReturnFnParamsVec,0); % [N_a,N_a,N_semiz], aprime-by-a-by-semiz

            entireRHS_e=ReturnMatrix_d2e+DiscountedEV_d2; % autoexpands over a

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V_ford2(:,:,e_c,d2_c)=reshape(Vtemp,[N_a,N_semiz]);
            Policy_ford2(:,:,e_c,d2_c)=reshape(maxindex,[N_a,N_semiz]);
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_semiz,N_e]
aprime_ind=reshape(Policy_ford2((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(reshape(maxindex,[N_a*N_semiz*N_e,1])-1)),[1,N_a,N_semiz,N_e]);
Policy=[reshape(maxindex,[1,N_a,N_semiz,N_e]); aprime_ind]; % [d2; aprime]


end
