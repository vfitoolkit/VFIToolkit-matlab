function [V,Policy]=ValueFnIter_FHorz_SemiExo_nod1_e_raw(n_d2,n_a,n_z,n_semiz, n_e,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d=prod(n_d2); % Needed at end to reshape output
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(2,N_a,N_semiz*N_z,N_e,N_j,'gpuArray'); % d2, aprime

%%
special_n_d2=ones(1,length(n_d2));

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_e=ones(1,length(n_e));
end
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
elseif vfoptions.lowmemory==3 % joint bothz, inner e
    V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
    Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, n_bothz, n_e, d2_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy(1,:,:,:,N_j)=shiftdim(d_ind,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, n_bothz, special_n_e, d2_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy(1,:,:,e_c,N_j)=shiftdim(d_ind,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, [n_semiz,special_n_z], special_n_e, d2_gridvals, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,semizblock,e_c,N_j)=Vtemp;
                Policy(1,:,semizblock,e_c,N_j)=rem(maxindex-1,N_d)+1;
                Policy(2,:,semizblock,e_c,N_j)=ceil(maxindex/N_d);
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d2, n_a, special_n_bothz, special_n_e, d2_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
                Policy(1,:,z_c,e_c,N_j)=shiftdim(d_ind,-1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
            end
        end
    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form
    EV=sum(EV.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_bothz, n_e, d2_val, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
            % (d,aprime,a,z,e)

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV_d2; %*repmat(entireEV,1,N_a,1,N_e);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d2e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_bothz, special_n_e, d2_val, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
                % (d,aprime,a,z)

                entireRHS_d2e=ReturnMatrix_d2e+DiscountFactorParamsVec*EV_d2; %.*ones(1,N_a,1);

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d2e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, [n_semiz,special_n_z], special_n_e, d2_val, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                    entireRHS_ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_d2z;
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, special_n_bothz, special_n_e, d2_val, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                    entireRHS_ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_d2z;
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,z_c,e_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=maxindex;
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,N_j)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
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

    EV=V(:,:,:,jj+1);
    EV=sum(EV.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj));

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_bothz, n_e, d2_val, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
            % (d,aprime,a,z,e)

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*EV_d2; %repmat(entireEV,1,N_a,1,N_e);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            EV_d2=EV.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, n_bothz, special_n_e, d2_val, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);
                % (d,aprime,a,z)

                entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV_d2; %.*ones(1,N_a,1);

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==2 % outer z / inner e, vectorize semiz
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, [n_semiz,special_n_z], special_n_e, d2_val, a_grid, z_valblock, e_val, ReturnFnParamsVec,0);
                    entireRHS_ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_d2z;
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(Vtemp,1);
                    Policy_ford2_jj(:,semizblock,e_c,d2_c)=shiftdim(maxindex,1);
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                EV_d2z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d2ze=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d2, n_a, special_n_bothz, special_n_e, d2_val, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                    entireRHS_ze=ReturnMatrix_d2ze+DiscountFactorParamsVec*EV_d2z;
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford2_jj(:,z_c,e_c,d2_c)=Vtemp;
                    Policy_ford2_jj(:,z_c,e_c,d2_c)=maxindex;
                end
            end
        end
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        Policy(2,:,:,:,jj)=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);

    end
end



end
