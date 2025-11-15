function [V,Policy]=ValueFnIter_FHorz_SemiExo_nod1_noz_raw(n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(2,N_a,N_semiz,N_j,'gpuArray'); % just d2 and aprime

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d2, n_a, n_semiz, d2_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(1,:,:,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
        Policy(2,:,:,N_j)=shiftdim(ceil(maxindex/N_d2),-1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d2, n_a, special_n_semiz, d2_gridvals, a_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(1,:,z_c,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
            Policy(2,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d2),-1);
        end

    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            % (d,aprime,a,z)

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV_d2; %repmat(EV,1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec);

                % Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_d2z=sum(EV_d2z,2);

                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_d2z; % EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);
            % (d,aprime,a,z)

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV_d2; %repmat(EV,1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_d2z=EV.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_d2z=sum(EV_d2z,2);

                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_d2z; % EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        
    end
end


end
