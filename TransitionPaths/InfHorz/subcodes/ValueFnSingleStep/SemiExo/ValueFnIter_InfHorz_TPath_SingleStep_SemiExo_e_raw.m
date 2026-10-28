function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_e_raw(Vnext,n_d1,n_d2,n_a,n_z,n_semiz, n_e, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, e_gridvals, pi_z, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with semi-exogenous state semiz, with d1 and d2 (d2 determines the transition of semiz), with z and iid e.
% Vnext is [N_a,N_bothz,N_e] (next period value fn), bothz=(semiz,z) with semiz first
% pi_z is [N_z,N_zprime]; pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition); pi_e is [N_e,1], the distribution of next period e.
% Policy is returned as kron rows [d1; d2; aprime], [3,N_a,N_bothz,N_e]
% lowmemory: 0 is everything at once; 1 loops over e.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

special_n_d=[n_d1,ones(1,length(n_d2))];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

bothz_gridvals=[repmat(semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];

% Preallocate
V_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_bothz,N_e,N_d2,'gpuArray');

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then z', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_bothz,N_e]).*reshape(pi_e,[1,1,N_e]); % [N_a,N_bothz,N_e], aprime-by-(semizprime,zprime)-by-eprime
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_bothz]
EV=reshape(EV,[N_a,N_semiz,N_z]).*shiftdim(pi_z',-2); % [N_a,N_semiz,N_zprime,N_z], aprime-by-semizprime-by-zprime-by-z
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,3); % sum over z', [N_a,N_semiz,1,N_z], aprime-by-semizprime-by-1-by-z
EV=EV.*permute(pi_semiz,[4,2,1,5,3]); % [N_a,N_semizprime,N_semiz,N_z,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_z,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_bothz,N_d2]); % aprime-by-(semiz,z)-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec,0); % [N_d1*N_a,N_a,N_bothz,N_e], (d1,aprime)-by-a-by-(semiz,z)-by-e

        entireRHS=ReturnMatrix_d2+repelem(reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_bothz]),N_d1,1,1); % autoexpands over a and e

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V_ford2(:,:,:,d2_c)=reshape(Vtemp,[N_a,N_bothz,N_e]);
        Policy_ford2(:,:,:,d2_c)=reshape(maxindex,[N_a,N_bothz,N_e]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d12c_gridvals=d12_gridvals(:,:,d2_c);
        entireEV=repelem(reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_bothz]),N_d1,1,1); % [N_d1*N_a,1,N_bothz]

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals, e_val, ReturnFnParamsVec,0); % [N_d1*N_a,N_a,N_bothz], (d1,aprime)-by-a-by-(semiz,z)

            entireRHS_e=ReturnMatrix_d2e+entireEV; % autoexpands over a

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V_ford2(:,:,e_c,d2_c)=reshape(Vtemp,[N_a,N_bothz]);
            Policy_ford2(:,:,e_c,d2_c)=reshape(maxindex,[N_a,N_bothz]);
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],4); % max over d2, [N_a,N_bothz,N_e]
d1aprime_ind=reshape(Policy_ford2((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(reshape(maxindex,[N_a*N_bothz*N_e,1])-1)),[1,N_a,N_bothz,N_e]);
Policy=[rem(d1aprime_ind-1,N_d1)+1; reshape(maxindex,[1,N_a,N_bothz,N_e]); ceil(d1aprime_ind/N_d1)]; % [d1; d2; aprime]


end
