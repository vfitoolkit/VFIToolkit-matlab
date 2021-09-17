function [V, Policyhat]=ValueFnIter_Case1_FHorz_SQHyperbolic_SingleStep_no_d_raw(V,n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Note, with Sophisticated QuasiHyperbolic V contains both Vunderbar and Vhat, the
% exponential discount of the sophisticated policy and sophisticated quasi-hyperbolic discounter
% respectively, while Policyhat is the sophisticated quasi-hyperbolic discounter.

% V.Vunderbar
% V.Vhat

% % Quasi-hyperbolic discount factors
% beta=prod(DiscountFactorParamsVec(1:end-1));
% beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period

N_a=prod(n_a);
N_z=prod(n_z);

Policyhat=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')


if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V.Vunderbar(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end

if vfoptions.lowmemory==0
    % In the final age period there is no discounting, so Vunderbar and Vhat are just the same thing
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V.Vhat(:,:,N_j)=Vtemp;
    Policyhat(:,:,N_j)=maxindex;
    V.Vunderbar(:,:,N_j)=Vtemp;

elseif vfoptions.lowmemory==1
    
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V.Vhat(:,z_c,N_j)=Vtemp;
        Policyhat(:,z_c,N_j)=maxindex;
        V.Vunderbar(:,z_c,N_j)=Vtemp;
    end
    
elseif vfoptions.lowmemory==2

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V.Vhat(a_c,z_c,N_j)=Vtemp;
            Policyhat(a_c,z_c,N_j)=maxindex;
            V.Vunderbar(a_c,z_c,N_j)=Vtemp;

        end
    end   
    
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        sprintf('Finite horizon: %i of %i (counting backwards to 1)',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec(1:end-1));
    beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period

    if fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    
    VKronNext_j=Vtemp_j; % Has been presaved before it was VKronNext_j=V(:,:,j+1);
    Vtemp_j=V.Vunderbar(:,:,jj); % Grab this before it is replaced/updated
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);

        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Vhat and Policyhat
            entireRHS_z=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V.Vhat(:,z_c,jj)=Vtemp;
            Policyhat(:,z_c,jj)=maxindex;
            % Now use Policyhat to calculate Vunderbar
            entireRHS_z=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1);
            tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
            V.Vunderbar(:,z_c,jj)=entireRHS_z(tempmaxindex);
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Vhat and Policyhat
            entireRHS_z=ReturnMatrix_z+beta0beta*EV_z*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V.Vhat(:,z_c,jj)=Vtemp;
            Policyhat(:,z_c,jj)=maxindex;
            % Now use Policyhat to calculate Vunderbar
            entireRHS_z=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1);
            tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
            V.Vunderbar(:,z_c,jj)=entireRHS_z(tempmaxindex);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
                        
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_z
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                
                % Vhat and Policyhat
                entireRHS_az=ReturnMatrix_az+beta0beta*EV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V.Vhat(a_c,z_c,jj)=Vtemp;
                Policyhat(a_c,z_c,jj)=maxindex;
                % Now use Policyhat to calculate Vunderbar
                entireRHS_az=ReturnMatrix_az+beta*EV_z;
                V.Vunderbar(a_c,z_c,jj)=entireRHS_az(maxindex);
            end
        end
        
    end
end


end