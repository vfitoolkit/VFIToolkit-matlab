function [V,Policy2]=ValueFnIter_Case1_FHorz_Ambiguity_e_raw(n_ambiguity,n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_grid_J, e_grid_J,ambiguity_pi_z_J, ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    % z_gridvals is created below
end

ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-2); % Move to third dimension for e_c=1:n_e

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid_J(:,N_j), e_grid_J(:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_grid_J(e_c,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid_J(:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_grid_J(e_c,N_j);
            for z_c=1:N_z
                z_val=z_grid_J(z_c,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    ambEV=zeros(N_a,n_z,n_ambiguity(N_j)); % aprime,zprime, prior
    for amb_c=1:n_ambiguity(N_j) % Evaluate expections under each of the multiple priors
        EV=V_Jplus1.*ambiguity_pi_e_J(1,1,:,N_j,amb_c);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,3); % sum over e, leaving a singular third dimension
        ambEV(:,:,amb_c)=EV;
    end
    % Take the worst-case over the priors
    EV=min(ambEV,[],3);
    % We will then use this to take expectation in z'

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid_J(:,N_j), e_grid_J(:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z,e)

        ambEV=zeros(N_a,1,N_z,n_ambiguity(N_j)); % aprime, nothing, z, prior
        for amb_c=1:n_ambiguity(N_j) % Evaluate expections under each of the multiple priors
            EV=EV.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEV(:,:,:,amb_c)=EV;
        end
        % Take the worst-case over the priors
        EV=min(ambEV,[],4);
        % From here, can just use EV as normal
        
        entireEV=repelem(EV,N_d,1,1);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        ambEV=zeros(N_a,1,N_z,n_ambiguity(N_j)); % aprime, nothing, z, prior
        for amb_c=1:n_ambiguity(N_j) % Evaluate expections under each of the multiple priors
            EV=EV.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEV(:,:,:,amb_c)=EV;
        end
        % Take the worst-case over the priors
        EV=min(ambEV,[],4);
        % From here, can just use EV as normal

        entireEV=repelem(EV,N_d,1,1);

        for e_c=1:N_e
            e_val=e_grid_J(e_c,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid_J(:,N_j), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV.*ones(1,N_a,1);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_grid_J(z_c,N_j);
            
            ambEV_z=zeros(N_a,n_ambiguity(N_j)); % aprime, prior
            for amb_c=1:n_ambiguity(N_j) % Evaluate expections under each of the multiple priors
                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*ambiguity_pi_z_J(z_c,:,N_j,amb_c));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                ambEV_z(:,amb_c)=EV_z;
            end
            % Take the worst-case over the priors
            EV_z=min(ambEV_z,[],2);
            % From here, can just use EV_z as normal
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_grid_J(e_c,N_j);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
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
    
    ambEV=zeros(N_a,n_z,n_ambiguity(jj)); % aprime,zprime, prior
    for amb_c=1:n_ambiguity(jj) % Evaluate expections under each of the multiple priors
        EV=VKronNext_j.*ambiguity_pi_e_J(1,1,:,jj,amb_c);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,3); % sum over e, leaving a singular third dimension
        ambEV(:,:,amb_c)=EV;
    end
    % Take the worst-case over the priors
    EV=min(ambEV,[],3);
    % We will then use this to take expectation in z'

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid_J(:,jj), e_grid_J(:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        ambEV=zeros(N_a,1,N_z,n_ambiguity(jj)); % aprime, nothing, z, prior
        for amb_c=1:n_ambiguity(jj) % Evaluate expections under each of the multiple priors
            EV=EV.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEV(:,:,:,amb_c)=EV;
        end
        % Take the worst-case over the priors
        EV=min(ambEV,[],4);
        % From here, can just use EV as normal

        entireEV=repelem(EV,N_d,1,1);
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        ambEV=zeros(N_a,1,N_z,n_ambiguity(jj)); % aprime, nothing, z, prior
        for amb_c=1:n_ambiguity(jj) % Evaluate expections under each of the multiple priors
            EV=EV.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEV(:,:,:,amb_c)=EV;
        end
        % Take the worst-case over the priors
        EV=min(ambEV,[],4);
        % From here, can just use EV as normal

        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_grid_J(e_c,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid_J(:,jj), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV.*ones(1,N_a,1);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_grid_J(z_c,jj);
            
            ambEV_z=zeros(N_a,n_ambiguity(jj)); % aprime, prior
            for amb_c=1:n_ambiguity(jj) % Evaluate expections under each of the multiple priors
                %Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*ambiguity_pi_z_J(z_c,:,jj,amb_c));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                ambEV_z(:,amb_c)=EV_z;
            end
            % Take the worst-case over the priors
            EV_z=min(ambEV_z,[],2);
            % From here, can just use EV_z as normal
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_grid_J(e_c,jj);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
    end

end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end