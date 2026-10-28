function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_noz_raw(Vnext,n_d1,n_d2,n_a,n_semiz, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with semi-exogenous state semiz, with d1 and d2 (d2 determines the transition of semiz), no z.
% Vnext is [N_a,N_semiz] (next period value fn)
% pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition)
% Policy is returned as kron rows [d1; d2; aprime], [3,N_a,N_semiz]
% lowmemory: 0 is everything at once; 1 loops over semiz.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_semiz,N_d2,'gpuArray');

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz]).*permute(pi_semiz,[4,2,1,3]); % [N_a,N_semizprime,N_semiz,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_semiz,N_d2]); % aprime-by-semiz-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals, ReturnFnParamsVec,0); % [N_d1*N_a,N_a,N_semiz], (d1,aprime)-by-a-by-semiz

        entireRHS=ReturnMatrix_d2+repelem(reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_semiz]),N_d1,1,1); % the N_a autofills

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V_ford2(:,:,d2_c)=reshape(Vtemp,[N_a,N_semiz]);
        Policy_ford2(:,:,d2_c)=reshape(maxindex,[N_a,N_semiz]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        for z_c=1:N_semiz
            z_val=semiz_gridvals(z_c,:);
            ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,0); % [N_d1*N_a,N_a], (d1,aprime)-by-a

            entireRHS_z=ReturnMatrix_d2z+repelem(DiscountedEV(:,z_c,d2_c),N_d1,1); % the N_a autofills

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V_ford2(:,z_c,d2_c)=reshape(Vtemp,[N_a,1]);
            Policy_ford2(:,z_c,d2_c)=reshape(maxindex,[N_a,1]);
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],3); % max over d2, [N_a,N_semiz]
d1aprime_ind=reshape(Policy_ford2((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(reshape(maxindex,[N_a*N_semiz,1])-1)),[1,N_a,N_semiz]);
Policy=[rem(d1aprime_ind-1,N_d1)+1; reshape(maxindex,[1,N_a,N_semiz]); ceil(d1aprime_ind/N_d1)]; % [d1; d2; aprime]


end
