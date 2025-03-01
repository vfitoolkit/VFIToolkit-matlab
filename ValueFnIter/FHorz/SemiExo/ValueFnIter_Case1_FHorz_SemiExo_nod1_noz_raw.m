function [V,Policy3]=ValueFnIter_Case1_FHorz_SemiExo_nod1_noz_raw(n_d2,n_a,n_semiz,N_j, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if isfield(vfoptions,'customahr')
    if vfoptions.customahr==1
        % NOTE: Following function is NOT part of VFI Toolkit
        [V,Policy3]=ValueFnIter_Case1_FHorz_SemiExo_nod1_noz_CUSTOM_raw(n_d2,n_a,n_semiz,N_j, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        return
    end
end

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(2,N_a,N_semiz,N_j,'gpuArray'); % just d2 and aprime

%%
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

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

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d2, n_a, n_semiz, d2_grid, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy3(1,:,:,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(maxindex/N_d2),-1);

    elseif vfoptions.lowmemory==1

        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d2, n_a, special_n_semiz, d2_grid, a_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy3(1,:,z_c,N_j)=shiftdim(rem(maxindex-1,N_d2)+1,-1);
            Policy3(2,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d2),-1);
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d2_val=d2_gridvals(d2_c,:)';

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            % (d,aprime,a,z)

            if vfoptions.paroverz==1

                EV=V_Jplus1.*shiftdim(pi_semiz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV; %repmat(EV,1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);


            elseif vfoptions.paroverz==0

                for z_c=1:N_semiz
                    ReturnMatrix_d2z=ReturnMatrix_d2(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_z; % EV_z*ones(1,N_a,1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V_ford2_jj(:,z_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,d2_c)=maxindex;
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy3(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d2_val=d2_gridvals(d2_c,:)';

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_z; % EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy3(1,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy3(2,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

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
    
    VKronNext_j=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d2_val=d2_gridvals(d2_c,:)';

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, n_semiz, d2_val, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);
            % (d,aprime,a,z)

            

            if vfoptions.paroverz==1

                EV=VKronNext_j.*shiftdim(pi_semiz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV; %repmat(EV,1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            elseif vfoptions.paroverz==0

                for z_c=1:N_semiz
                    ReturnMatrix_d2z=ReturnMatrix_d2(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_z; % EV_z*ones(1,N_a,1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_z,[],1);
                    V_ford2_jj(:,z_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,d2_c)=maxindex;
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy3(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy3(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d2_val=d2_gridvals(d2_c,:)';

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, special_n_d2, n_a, special_n_semiz, d2_val, a_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_semiz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*EV_z; % EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy3(1,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy3(2,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
        
    end
end


end
