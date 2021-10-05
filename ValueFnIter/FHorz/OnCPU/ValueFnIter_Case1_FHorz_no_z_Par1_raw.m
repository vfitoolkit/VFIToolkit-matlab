function [V,Policy2]=ValueFnIter_Case1_FHorz_no_z_Par1_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j);
Policy=zeros(N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
a_grid=gather(a_grid);
z_grid=gather(z_grid);

% eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    z_val=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel,ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2

    parfor a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, 0,ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_az);
        V(a_c,1,N_j)=Vtemp;
        Policy(a_c,1,N_j)=maxindex;        
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
    
    VKronNext_j=V(:,:,j+1);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        
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
    
        entireEV_z=kron(VKronNext_j,ones(N_d,1));
        entireRHS_z=ReturnMatrix(:,:,1);+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
    
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,1,j)=Vtemp;
        Policy(:,1,j)=maxindex;
    
    elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
        
        entireEV_z=kron(VKronNext_j,ones(N_d,1));
        
        parfor a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, 0,ReturnFnParamsVec);
            
            entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*entireEV_z;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_az);
            V(a_c,1,j)=Vtemp;
            Policy(a_c,1,j)=maxindex;
        end
    end
    
end

%%
Policy2=zeros(2,N_a,N_z,N_j); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end