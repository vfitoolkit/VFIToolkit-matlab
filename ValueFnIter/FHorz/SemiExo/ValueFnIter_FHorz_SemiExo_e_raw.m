function [V,Policy3]=ValueFnIter_FHorz_SemiExo_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]); % Needed at end to reshape output
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_z)+length(n_semiz));
end
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
elseif vfoptions.lowmemory==2 % loops over e and z
    V_ford2_jj=zeros(N_a,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_d2,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, n_e, [d1_grid; d2_grid], a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, n_bothz, special_n_e, [d1_grid; d2_grid], a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy3(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy3(3,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,n_d2], n_a, special_n_bothz, special_n_e, [d1_grid; d2_grid], a_grid, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
                Policy3(1,:,z_c,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
                Policy3(2,:,z_c,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
                Policy3(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, n_e, [d1_grid; d2_gridvals(d2_c,:)'], a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
            % (d,aprime,a,z,e)

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repelem(EV,N_d1,1,1);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*entireEV; %*repmat(entireEV,1,N_a,1,N_e);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=rem(d1aprime_ind-1,N_d1)+1;
        Policy3(3,:,:,:,N_j)=ceil(d1aprime_ind/N_d1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d2_val=d2_gridvals(d2_c,:)';

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repelem(EV,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, special_n_e, [d1_grid;d2_val], a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
                % (d,aprime,a,z)

                entireRHS_d2e=ReturnMatrix_d2e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d2e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

    elseif vfoptions.lowmemory==2
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
            d2_val=d2_gridvals(d2_c,:)';

            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,N_j);

                %Calc the condl expectation term (except beta) which depends on z but not control variables
                EV_z=V_Jplus1.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                entireEV_z=kron(EV_z,ones(N_d1,1));

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, special_n_bothz, special_n_e, [d1_grid;d2_val], a_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_d2ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d2ze,[],1);
                    
                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
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
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, n_e, [d1_grid; d2_gridvals(d2_c,:)'], a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
            % (d,aprime,a,z,e)

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repelem(EV,N_d1,1,1);
            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*entireEV; %repmat(entireEV,1,N_a,1,N_e);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end

        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repelem(EV,N_d1,1,1);

            d2_val=d2_gridvals(d2_c,:)';

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, n_bothz, special_n_e, [d1_grid;d2_val], a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
                % (d,aprime,a,z)

                entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
        
    elseif vfoptions.lowmemory==2
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            d2_val=d2_gridvals(d2_c,:)';
            
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                %Calc the condl expectation term (except beta) which depends on z but not control variables
                EV_z=VKronNext_j.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                entireEV_z=kron(EV_z,ones(N_d1,1));

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, [n_d1,special_n_d2], n_a, special_n_bothz, special_n_e, [d1_grid;d2_val], a_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                    V_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
        Policy3(1,:,:,:,jj)=reshape(rem(d1aprime_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
        Policy3(3,:,:,:,jj)=reshape(ceil(d1aprime_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
    end

end



end
