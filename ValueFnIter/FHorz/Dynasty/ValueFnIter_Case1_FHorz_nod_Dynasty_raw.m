function [V, Policy]=ValueFnIter_Case1_FHorz_nod_Dynasty_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end


%%
Vold=zeros(N_a,N_z,N_j);
tempcounter=1;
currdist=Inf;
while currdist>vfoptions.tolerance
    %% Iterate backwards through j.
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;
        
        if vfoptions.verbose==1
            sprintf('Finite horizon: %i of %i (counting backwards to 1)',jj, N_j)
        end
        
        
        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
        
        if reverse_j==0 % So j==N_j
            VKronNext_j=V(:,:,1);
        else
            VKronNext_j=V(:,:,jj+1);
        end
        
        if vfoptions.lowmemory==0
            
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
            
        elseif vfoptions.lowmemory==1
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,jj);
                ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
            
        elseif vfoptions.lowmemory==2
            for z_c=1:N_z
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                z_val=z_gridvals_J(z_c,:,jj);
                for a_c=1:N_a
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
    
    Vdist=reshape(V-Vold,[N_a*N_z*N_j,1]); Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?
    Vold=V;
    
    tempcounter=tempcounter+1;
    if vfoptions.verbose==1 && rem(tempcounter,10)==0
        fprintf('Value Fn Iteration: After %d steps, current distance is %8.2f \n', tempcounter, currdist);
    end
end


end