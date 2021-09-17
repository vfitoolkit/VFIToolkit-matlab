function [V,Policy2]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_raw(V,n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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

%% Set up the Epstein-Zin parameters (it is allowed for them to be age dependent)
DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(:,1:end-2),2),DiscountFactorParamsVec(:,end-1),DiscountFactorParamsVec(:,end)];
DiscountFactorParamsVec=DiscountFactorParamsVec';


%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
% Modify the Return Function appropriately for Epstein-Zin Preferences
% Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
% skip this and alter the (1-beta) term appropriately. Further, as this
% is just multiplying by a constant nor will it effect the argmax, so
% can just scale solution to the max directly.

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
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=((1-DiscountFactorParamsVec(jj,1)).^(1/(1-1/DiscountFactorParamsVec(jj,3))))*Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=((1-DiscountFactorParamsVec(jj,1)).^(1/(1-1/DiscountFactorParamsVec(jj,3))))*Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2

    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V(a_c,z_c,N_j)=((1-DiscountFactorParamsVec(jj,1)).^(1/(1-1/DiscountFactorParamsVec(jj,3))))*Vtemp;
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
%     DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,j);
%     DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

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
    
    VKronNext_j=Vtemp_j; % Has been presaved before it was
%     VKronNext_j=V(:,:,j+1);
    Vtemp_j=V(:,:,j); % Grab this before it is replaced/updated

    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(jj,3));
        temp2=(1-DiscountFactorParamsVec(jj,1))*temp2;
        
%         %Calc the condl expectation term (except beta), which depends on z but
%         %not on control variables
%         EV=VKronNext_j.*(ones(N_a,1,'gpuArray')*dimshift(pi_z,1)); %THIS LINE IS LIKELY INCORRECT
%         EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%         EV=sum(EV,2);
%         
%         entireEV=kron(EV,ones(N_d,1,1));
%         entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV*ones(1,N_a,N_z);
%         
%         %Calc the max and it's index
%         [Vtemp,maxindex]=max(entireRHS,[],3);
%         V(:,:,j)=Vtemp;
%         PolicyIndexes(:,:,j)=maxindex;

         for z_c=1:N_z
            ReturnMatrix_z=temp2(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Could prob do all this without creating temp3, just operate
            % direct on EV_z, as last third line is just
            % EV_z(isnan(EV_z))=0; to deal with NaN resulting from 0 to
            % negative power (as earlier, replace them with zeros (as the
            % zeros come from the transition probabilities)
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(:,3))/(1-DiscountFactorParamsVec(:,2)));
            temp3(EV_z==0)=0;
            
            entireEV_z=kron(temp3,ones(N_d,1));
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec(jj,1)*entireEV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,j)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(jj,3)));
            Policy(:,z_c,j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2_z=ReturnMatrix_z;
            temp2_z(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(jj,3));
            temp2_z=(1-DiscountFactorParamsVec(jj,1))*temp2_z;
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Could prob do all this without creating temp3, just operate
            % direct on EV_z, as last third line is just
            % EV_z(isnan(EV_z))=0; to deal with NaN resulting from 0 to
            % negative power (as earlier, replace them with zeros (as the
            % zeros come from the transition probabilities)
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(:,3))/(1-DiscountFactorParamsVec(:,2)));
            temp3(EV_z==0)=0;
            
            entireEV_z=kron(temp3_z,ones(N_d,1));
            entireRHS_z=temp2_z+DiscountFactorParamsVec(jj,1)*entireEV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,j)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(jj,3)));
            Policy(:,z_c,j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            % Could prob do all this without creating temp3, just operate
            % direct on EV_z, as last third line is just
            % EV_z(isnan(EV_z))=0; to deal with NaN resulting from 0 to
            % negative power (as earlier, replace them with zeros (as the
            % zeros come from the transition probabilities)
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(:,3))/(1-DiscountFactorParamsVec(:,2)));
            temp3(EV_z==0)=0;
            
            entireEV_z=kron(temp3,ones(N_d,1));
            
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_z
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
                
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                temp2_az=ReturnMatrix_az;
                temp2_az(isfinite(ReturnMatrix_az))=ReturnMatrix_az(isfinite(ReturnMatrix_az)).^(1-1/DiscountFactorParamsVec(jj,3));
                temp2_az=(1-DiscountFactorParamsVec(jj,1))*temp2_az;
                
                entireRHS_az=temp2_az+DiscountFactorParamsVec(jj,1)*entireEV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,j)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(jj,3)));
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