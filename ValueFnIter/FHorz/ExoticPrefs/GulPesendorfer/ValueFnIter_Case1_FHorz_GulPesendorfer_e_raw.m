function [V,Policy2]=ValueFnIter_Case1_FHorz_GulPesendorfer_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if n_z(1)==0
    l_z=0;
else
    l_z=length(n_z);
end

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

special_n_e=ones(1,length(n_e)); % if vfoptions.lowmemory>0
pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);

                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
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
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=V_Jplus1.*shiftdim(pi_z_J',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*entireEV; %*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=V_Jplus1.*shiftdim(pi_z_J',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)

            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);
                        
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,N_j));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
                
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z_J',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*entireEV; %*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_gridvals_J(:,:,jj), e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e+DiscountFactorParamsVec*entireEV; %.*ones(1,N_a,1);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
                
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