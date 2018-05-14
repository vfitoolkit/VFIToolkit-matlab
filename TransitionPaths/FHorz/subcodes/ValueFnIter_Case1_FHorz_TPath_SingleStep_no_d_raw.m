function [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_no_d_raw(V,n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')


if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
    
    z_gridvals=zeros(N_z,length(n_z),'gpuArray');
    for i1=1:N_z
        sub=zeros(1,length(n_z));
        sub(1)=rem(i1-1,n_z(1))+1;
        for ii=2:length(n_z)-1
            sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
        end
        sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
        
        if length(n_z)>1
            sub=sub+[0,cumsum(n_z(1:end-1))];
        end
        z_gridvals(i1,:)=z_grid(sub);
    end
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    
    a_gridvals=zeros(N_a,length(n_a),'gpuArray');
    for i2=1:N_a
        sub=zeros(1,length(n_a));
        sub(1)=rem(i2-1,n_a(1)+1);
        for ii=2:length(n_a)-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
        
        if length(n_a)>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=a_grid(sub);
    end
end

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

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
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2

    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V(a_c,z_c,N_j)=Vtemp;
            Policy(a_c,z_c,N_j)=maxindex;

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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

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
    
    VKronNext_j=Vtemp_j; % Has been presaved before it was
%     VKronNext_j=V(:,:,j+1);
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated

    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % IN PRINCIPLE, WHY BOTHER TO LOOP OVER z AT ALL TO CALCULATE
        % entireRHS?? CAN IT BE VECTORIZED DIRECTLY?
%         %Calc the condl expectation term (except beta), which depends on z but
%         %not on control variables
%         EV=VKronNext_j*pi_z'; %THIS LINE IS LIKELY INCORRECT
%         EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%         %EV=sum(EV,2);
%         
%         entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV*ones(1,N_a,N_z);
%         
%         %Calc the max and it's index
%         [Vtemp,maxindex]=max(entireRHS,[],1);
%         V(:,:,j)=Vtemp;
%         Policy(:,:,j)=maxindex;

        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
                        
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_z
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                
                entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*EV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end

% %%
% for reverse_j=1:N_j-1
%     j=N_j-reverse_j;
%     VKronNext_j=V(:,:,j+1);
%     FmatrixKron_j=reshape(FmatrixFn_j(j),[N_a,N_a,N_z]);
%     for z_c=1:N_z
%         RHSpart2=VKronNext_j.*kron(ones(N_a,1),pi_z(z_c,:));
%         RHSpart2(isnan(RHSpart2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%         RHSpart2=sum(RHSpart2,2);
%         for a_c=1:N_a
%             entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
%             
%             %calculate in order, the maximizing aprime indexes
%             [V(a_c,z_c,j),PolicyIndexes(1,a_c,z_c,j)]=max(entireRHS,[],1);
%         end
%     end
% end

end