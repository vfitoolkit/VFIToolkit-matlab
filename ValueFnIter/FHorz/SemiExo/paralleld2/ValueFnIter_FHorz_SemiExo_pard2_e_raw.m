function [V,Policy]=ValueFnIter_FHorz_SemiExo_pard2_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

n_d=[n_d1,n_d2];
d_grid=[d1_grid; d2_grid];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_z)+length(n_semiz));
end
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% pi_bothz_J=repmat(permute(pi_semiz_J,[3,2,1,4]),1,N_z,N_z,1).*repelem(permute(pi_z_J,[4,2,1,3]),1,N_semiz,N_semiz,1);
pi_semiz_J_permute=permute(pi_semiz_J,[3,2,1,4]); % (d2,semizprime,semiz,j)
pi_z_J_permute=permute(pi_z_J,[4,2,1,3]);% (1,zprime,z,j)

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz*N_z
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_bothz, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
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

        pi_bothz=repmat(pi_semiz_J_permute(:,:,:,N_j),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,N_j),1,N_semiz,N_semiz);
        % pi_bothz=pi_bothz_J(:,:,:,N_j);
        % (d2,zprime,z)

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d & aprime,a,z,e)

        EV=replem(V_Jplus1,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(EV,N_d1,1,1);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        pi_bothz=repmat(pi_semiz_J_permute(:,:,:,N_j),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,N_j),1,N_semiz,N_semiz);
        % (d2,zprime,z)

        EV=replem(V_Jplus1,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(EV,N_d1,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_grid, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % (d & aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);

            %Calc the condl expectation term (except beta) which depends on z but not control variables
            pi_bothz_z_c=repmat(pi_semiz_J_permute(:,:,z_c,N_j),1,N_z).*repelem(pi_z_J_permute(1,:,z_c,N_j),1,N_semiz);
            % (d2,zprime,1)

            EV_z=replem(V_Jplus1,N_d2,1).*repmat(pi_bothz_z_c,N_a,1,1);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension

            entireEV_z=repelem(EV_z,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_bothz, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                % (d & aprime,a,z)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
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
    % (aprime,zprime)

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, n_e, d_grid, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d & aprime,a,z,e)

        pi_bothz=repmat(pi_semiz_J_permute(:,:,:,jj),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,jj),1,N_semiz,N_semiz);

        EV=repelem(VKronNext_j,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1,1);

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,jj)=Vtemp;
        Policy(:,:,:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1

        pi_bothz=repmat(pi_semiz_J_permute(:,:,:,jj),1,N_z,N_z).*repelem(pi_z_J_permute(1,:,:,jj),1,N_semiz,N_semiz);
        timer(1)=toc;

        EV=replem(VKronNext_j,N_d2,1).*repmat(pi_bothz,N_a,1,1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireEV=repelem(EV,N_d1,1,1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_grid, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
            % (d & aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,jj)=Vtemp;
            Policy(:,:,e_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,jj);

            pi_bothz_z_c=repmat(pi_semiz_J_permute(:,:,z_c,jj),1,N_z).*repelem(pi_z_J_permute(1,:,z_c,jj),1,N_semiz);

            EV_z=replem(VKronNext_j,N_d2,1).*repmat(pi_bothz_z_c,N_a,1,1);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension

            entireEV_z=repelem(EV_z,N_d1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_bothz, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                % (d & aprime,a,z)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
    end

end



end