function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_nod_raw(n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DiscountFactorParamNames contains the names for the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [(1-beta)*u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function
%  psi is the elasticity of intertemporal solution
%  gamma is a measure of risk aversion, bigger gamma is more risk averse
%  beta is the standard marginal rate of time preference (discount factor)
%  When 1/(1-psi)=1-gamma, i.e., we get standard von-Neumann-Morgenstern

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1);
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
if length(DiscountFactorParamsVec)>3
    DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
end

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
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
        % skip this and alter the (1-beta) term appropriately.
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=((1-DiscountFactorParamsVec(1)).^(1/(1-1/DiscountFactorParamsVec(3))))*Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        %if vfoptions.returnmatrix==2 % GPU
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
            % skip this and alter the (1-beta) term appropriately. 
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=((1-DiscountFactorParamsVec(1)).^(1/(1-1/DiscountFactorParamsVec(3))))*Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        %if vfoptions.returnmatrix==2 % GPU
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
                % skip this and alter the (1-beta) term appropriately.
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_az);
                V(a_c,z_c,N_j)=((1-DiscountFactorParamsVec(1)).^(1/(1-1/DiscountFactorParamsVec(3))))*Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;

            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    VKronNext_j=V_Jplus1;
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));

        %Calc the expectation term (except beta)
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        % Use sparse for a few lines until sum over zprime
        EV=temp.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        temp3=EV;
        temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp3(EV==0)=0;
            
        entireRHS=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(1)*temp3*ones(1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
        
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2=ReturnMatrix_z;
            temp2(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));

            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EV_z==0)=0;
            
            entireRHS_z=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(3)*temp3*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        
        for z_c=1:N_z
            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EV_z==0)=0;
                        
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                temp2=ReturnMatrix_az;
                temp2(isfinite(ReturnMatrix_az))=ReturnMatrix_az(isfinite(ReturnMatrix_az)).^(1-1/DiscountFactorParamsVec(3));

                entireRHS_az=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(3)*temp3;
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az.^(1/(1-1/DiscountFactorParamsVec(3))));
                V(a_c,z_c,N_j)=Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;
            end
        end
        
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    if length(DiscountFactorParamsVec)>3
        DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
    end

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
    end
    
    
    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));

        %Calc the expectation term (except beta)
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        % Use sparse for a few lines until sum over zprime
        EV=temp.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        temp3=EV;
        temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp3(EV==0)=0;
            
        entireRHS=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(1)*temp3*ones(1,N_a,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        V(:,:,jj)=Vtemp;
        Policy(:,:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
        
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2=ReturnMatrix_z;
            temp2(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));

            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EV_z==0)=0;
            
            entireRHS_z=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(3)*temp3*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;
        
        for z_c=1:N_z
            %Calc the expectation term (except beta)
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp3=EV_z;
            temp3(isfinite(temp3))=temp3(isfinite(temp3)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
            temp3(EV_z==0)=0;
                        
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                temp2=ReturnMatrix_az;
                temp2(isfinite(ReturnMatrix_az))=ReturnMatrix_az(isfinite(ReturnMatrix_az)).^(1-1/DiscountFactorParamsVec(3));

                entireRHS_az=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(3)*temp3;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az.^(1/(1-1/DiscountFactorParamsVec(3))));
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end


end