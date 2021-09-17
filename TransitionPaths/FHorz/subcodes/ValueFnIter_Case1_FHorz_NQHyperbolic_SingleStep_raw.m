function [V,Policy2]=ValueFnIter_Case1_FHorz_NQHyperbolic_SingleStep_raw(V,n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Note, with Naive QuasiHyperbolic V contains both Vexp and VTilde, the
% exponential discounter and naive quasi-hyperbolic discounter
% respectively, while Policy is the naive quasi-hyperbolic discounter.

% V.Vexp
% V.Vtilde

% % Quasi-hyperbolic discount factors
% beta=prod(DiscountFactorParamsVec(1:end-1));
% beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V.Vexp(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

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
    % Exponential discounter
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V.Vexp(:,:,N_j)=Vtemp;
    % Naive Quasi-Hyperbolic discounter. Because this is the final period there is nothing to discount.
    % So just the same as the exponetial discounter.
    V.Vtilde(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    % Exponential discounter
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V.Vexp(:,z_c,N_j)=Vtemp;
        % Naive Quasi-Hyperbolic discounter. Because this is the final period there is nothing to discount.
        % So just the same as the exponetial discounter.
        V.Vtilde(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            % Exponential discounter
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V.Vexp(a_c,z_c,N_j)=Vtemp;
            % Naive Quasi-Hyperbolic discounter. Because this is the final period there is nothing to discount.
            % So just the same as the exponetial discounter.
            V.Vtilde(a_c,z_c,N_j)=Vtemp;
            Policy(a_c,z_c,N_j)=maxindex;
        end
    end   
    
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    j=N_j-reverse_j;

    if vfoptions.verbose==1
        sprintf('Finite horizon: %i of %i',j, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,j);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    beta=prod(DiscountFactorParamsVec(1:end-1));
    beta0beta=prod(DiscountFactorParamsVec); % Discount rate between present period and next period


    if fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,j);
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(j);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    
    VKronNext_j=Vtemp_j; % Has been presaved before it was VKronNext_j=V(:,:,j+1);
    Vtemp_j=V.Vexp(:,:,j); % Grab this before it is replaced/updated

    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            % Exponential discounter
            entireRHS_z=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
            [Vtemp,~]=max(entireRHS_z,[],1);
            V.Vexp(:,z_c,j)=Vtemp;
            % Naive Quasi-Hyperbolic discounter
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V.Vtilde(:,z_c,j)=Vtemp;
            Policy(:,z_c,j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            % Exponential discounter
            entireRHS_z=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
            [Vtemp,~]=max(entireRHS_z,[],1);
            V.Vexp(:,z_c,j)=Vtemp;
            % Naive Quasi-Hyperbolic discounter
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V.Vtilde(:,z_c,j)=Vtemp;
            Policy(:,z_c,j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_z
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);

                % Exponential discounter
                entireRHS_az=ReturnMatrix_az+beta*entireEV_z;
                [Vtemp,~]=max(entireRHS_az);
                V.Vexp(a_c,z_c,j)=Vtemp;
                % Naive Quasi-Hyperbolic discounter
                entireRHS_az=ReturnMatrix_az+beta0beta*entireEV_z;
                [Vtemp,maxindex]=max(entireRHS_az);
                V.Vtilde(a_c,z_c,j)=Vtemp;
                Policy(a_c,z_c,j)=maxindex;
            end
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end