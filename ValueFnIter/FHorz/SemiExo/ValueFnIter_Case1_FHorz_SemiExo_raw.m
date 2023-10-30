function [V,Policy3]=ValueFnIter_Case1_FHorz_SemiExo_raw(n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_grid, d2_grid, a_grid, z_grid, semiz_grid, pi_z, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod([n_d1,n_d2]); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);
semiz_grid=gpuArray(semiz_grid);
z_grid=gpuArray(z_grid);

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if vfoptions.lowmemory>0
    l_z=length(n_z);
    % z_gridvals is created below

    % The grid for semiz is not allowed to depend on age (the way the transition probabilities are calculated does not allow for it)
    if all(size(semiz_grid)==[sum(n_semiz),1])
        semiz_gridvals=CreateGridvals(n_semiz,semiz_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(semiz_grid)==[prod(n_semiz),l_semiz])
        semiz_gridvals=semiz_grid;
    end

    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if fieldexists_pi_z_J==1
    z_grid=vfoptions.z_grid_J(:,N_j);
    pi_z=vfoptions.pi_z_J(:,:,N_j);
elseif fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end
if vfoptions.lowmemory>0
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    bothz_gridvals=[kron(ones(N_semiz,1),z_gridvals),kron(semiz_gridvals,ones(N_z,1))];
else
    if all(size(z_grid)==[sum(n_z),1]) % if z are not using correlated/joint grid, then it is assumed semiz is not either
        bothz_grid=[semiz_grid; z_grid];
    elseif all(size(z_grid)==[prod(n_z),l_z])
        % Joint z_gridvals with semiz_gridvals (note that because z_grid is a joint/correlated grid z_gridvals is anyway just z_grid)
        bothz_grid=[kron(semiz_gridvals,ones(N_z,1)),kron(ones(N_semiz,1),z_grid)];
        bothz_gridvals=both_zgrid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, [n_d1,n_d2], n_a, n_bothz, [d1_grid; d2_grid], a_grid, bothz_grid, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
        Policy3(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy3(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        %if vfoptions.returnmatrix==2 % GPU
        for z_c=1:N_bothz
            z_val=bothz_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, [n_d1,n_d2], n_a, special_n_bothz, [d1_grid; d2_grid], a_grid, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=shiftdim(rem(maxindex-1,N_d)+1,-1);
            Policy3(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy3(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy3(3,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z, pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            %if vfoptions.returnmatrix==2 % GPU
            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, [n_d1,1], n_a, n_bothz, [d1_grid; d2_grid(d2_c)], a_grid, bothz_grid, ReturnFnParamsVec);
            % (d,aprime,a,z)

            if vfoptions.paroverz==1

                EV=V_Jplus1.*shiftdim(pi_bothz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                entireEV=kron(EV,ones(N_d1,1));
                %             entireEV=repelem(EV,N_d,1,1); % I tried this instead but appears repelem() is slower than kron()
                entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);


            elseif vfoptions.paroverz==0

                for z_c=1:N_bothz
                    ReturnMatrix_d2z=ReturnMatrix_d2(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    entireEV_z=kron(EV_z,ones(N_d1,1));
                    entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);

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
        Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, [n_d1,1], n_a, special_n_bothz, [d1_grid,d2_grid(d2_c)], a_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireEV_z=kron(EV_z,ones(N_d1,1));
                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,N_j)=V_jj;
        Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

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

    if fieldexists_pi_z_J==1
        z_grid=vfoptions.z_grid_J(:,jj);
        pi_z=vfoptions.pi_z_J(:,:,jj);
    elseif fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    if vfoptions.lowmemory>0 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(z_grid)==[prod(n_z),l_z])
            z_gridvals=z_grid;
        end
        bothz_gridvals=[kron(semiz_gridvals,ones(N_z,1)),kron(ones(N_semiz,1),z_gridvals)];
    else
        if all(size(z_grid)==[sum(n_z),1]) % if z are not using correlated/joint grid, then it is assumed semiz is not either
            bothz_grid=[semiz_grid; z_grid];
        elseif all(size(z_grid)==[prod(n_z),l_z])
            % Joint z_gridvals with semiz_gridvals (note that because z_grid is a joint/correlated grid z_gridvals is anyway just z_grid)
            bothz_grid=[kron(semiz_gridvals,ones(N_z,1)),kron(ones(N_semiz,1),z_grid)];
            bothz_gridvals=both_zgrid;
        end
    end
    
    VKronNext_j=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d2_c,jj)); % reverse order

            %if vfoptions.returnmatrix==2 % GPU
            ReturnMatrix_d2=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, [n_d1,1], n_a, n_bothz, [d1_grid;d2_grid(d2_c)], a_grid, bothz_grid, ReturnFnParamsVec);
            % (d,aprime,a,z)

            if vfoptions.paroverz==1

                EV=VKronNext_j.*shiftdim(pi_bothz',-1);
                EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                entireEV=kron(EV,ones(N_d1,1));
                %             entireEV=repelem(EV,N_d,1,1); % I tried this instead but appears repelem() is slower than kron()
                entireRHS=ReturnMatrix_d2+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            elseif vfoptions.paroverz==0

                for z_c=1:N_bothz
                    ReturnMatrix_d2z=ReturnMatrix_d2(:,:,z_c);

                    %Calc the condl expectation term (except beta), which depends on z but not on control variables
                    EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    EV_z=sum(EV_z,2);

                    entireEV_z=kron(EV_z,ones(N_d1,1));
                    entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);

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
        Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z,pi_semiz_J(:,:,d2_c,jj)); % reverse order

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);
                ReturnMatrix_d2z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d1, n_a, special_n_bothz, d_grid, a_grid, z_val, ReturnFnParamsVec);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_bothz(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireEV_z=kron(EV_z,ones(N_d1,1));
                entireRHS_z=ReturnMatrix_d2z+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford2_jj(:,z_c,d2_c)=Vtemp;
                Policy_ford2_jj(:,z_c,d2_c)=maxindex;
            end
        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
        V(:,:,jj)=V_jj;
        Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]); % This is the value of d that corresponds, make it this shape for addition just below
        d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
        Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
        Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);
        
    end
end


end