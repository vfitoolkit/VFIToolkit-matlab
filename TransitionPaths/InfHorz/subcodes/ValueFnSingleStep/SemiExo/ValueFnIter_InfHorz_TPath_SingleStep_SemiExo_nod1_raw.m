function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_nod1_raw(Vnext,n_d2,n_a,n_z,n_semiz, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, with semi-exogenous state semiz, no d1, with d2 (d2 determines the transition of semiz), with z.
% Vnext is [N_a,N_bothz] (next period value fn), bothz=(semiz,z) with semiz first
% pi_z is [N_z,N_zprime]; pi_semiz is [N_semiz,N_semizprime,N_d2] (this period's transition)
% Policy is returned as kron rows [d2; aprime], [2,N_a,N_bothz]
% lowmemory: 0 is everything at once; 1 loops over (semiz,z).

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

if vfoptions.lowmemory>=2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states')
end

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals=[repmat(semiz_gridvals,N_z,1),repelem(z_gridvals,N_semiz,1)];

% Preallocate
V_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');
Policy_ford2=zeros(N_a,N_bothz,N_d2,'gpuArray');

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First z', then semiz' (given d2)
EV=reshape(Vnext,[N_a,N_semiz,N_z]).*shiftdim(pi_z',-2); % [N_a,N_semiz,N_zprime,N_z], aprime-by-semizprime-by-zprime-by-z
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,3); % sum over z', [N_a,N_semiz,1,N_z], aprime-by-semizprime-by-1-by-z
EV=EV.*permute(pi_semiz,[4,2,1,5,3]); % [N_a,N_semizprime,N_semiz,N_z,N_d2]
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over semiz', [N_a,1,N_semiz,N_z,N_d2]
DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,N_bothz,N_d2]); % aprime-by-(semiz,z)-by-d2

%%
if vfoptions.lowmemory==0
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, n_bothz, d2_val, a_grid, bothz_gridvals, ReturnFnParamsVec,0); % [N_a,N_a,N_bothz], aprime-by-a-by-(semiz,z)

        entireRHS=ReturnMatrix_d2+reshape(DiscountedEV(:,:,d2_c),[N_a,1,N_bothz]); % the N_a autofills

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V_ford2(:,:,d2_c)=reshape(Vtemp,[N_a,N_bothz]);
        Policy_ford2(:,:,d2_c)=reshape(maxindex,[N_a,N_bothz]);
    end

elseif vfoptions.lowmemory==1
    for d2_c=1:N_d2
        d2_val=d2_gridvals(d2_c,:);

        for z_c=1:N_bothz
            z_val=bothz_gridvals(z_c,:);
            ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d2, n_a, special_n_bothz, d2_val, a_grid, z_val, ReturnFnParamsVec,0); % [N_a,N_a], aprime-by-a

            entireRHS_z=ReturnMatrix_d2z+DiscountedEV(:,z_c,d2_c); % the N_a autofills

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V_ford2(:,z_c,d2_c)=reshape(Vtemp,[N_a,1]);
            Policy_ford2(:,z_c,d2_c)=reshape(maxindex,[N_a,1]);
        end
    end
end

% Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
[V,maxindex]=max(V_ford2,[],3); % max over d2, [N_a,N_bothz]
aprime_ind=reshape(Policy_ford2((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(reshape(maxindex,[N_a*N_bothz,1])-1)),[1,N_a,N_bothz]);
Policy=[reshape(maxindex,[1,N_a,N_bothz]); aprime_ind]; % [d2; aprime]


end
