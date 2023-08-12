function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_no_d_raw(n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglowweight, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)
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
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc2=1-DiscountFactorParamsVec;
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

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
    % Modify the Return Function appropriately for Epstein-Zin Preferences

    %Calc the max and it's index
    [Vtemp,maxindex]=max((ezc2*ReturnMatrix.^ezc3).^ezc7,[],1);
    V(:,:,N_j)=ezc1*Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
        % Modify the Return Function appropriately for Epstein-Zin Preferences

        %Calc the max and it's index
        [Vtemp,maxindex]=max((ezc2*ReturnMatrix_z.^ezc3).^ezc7,[],1);
        V(:,z_c,N_j)=ezc1*Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2

    %if vfoptions.returnmatrix==2 % GPU
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences

            %Calc the max and it's index
            [Vtemp,maxindex]=max((ezc2*ReturnMatrix_az.^ezc3).^ezc7);
            V(a_c,z_c,N_j)=ezc1*Vtemp;
            Policy(a_c,z_c,N_j)=maxindex;

        end
    end   
    
end

disp('HERE')

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc2=1-DiscountFactorParamsVec;
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

    if jj==80
        max(max(abs(VKronNext_j)))
    end

    % Part of Epstein-Zin is before taking expectation
    temp=VKronNext_j;
    temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^ezc5;
    temp(VKronNext_j==0)=0;

    if jj==80
        max(max(abs(temp)))
    end

    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^ezc3;
        temp2(ReturnMatrix==0)=0; % matlab otherwise puts 0 to negative power to infinity

        for z_c=1:N_z
            temp2_z=temp2(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            temp4(isfinite(temp4))=(ezc4*sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV_z==0)=0;

            if jj==80
                z_c
                max(max(abs(temp2)))
                max(max(abs(temp4)))
            end
            
            entireRHS_z=ezc2*temp2_z+DiscountFactorParamsVec*temp4*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z.^ezc7,[],1);
            V(:,z_c,jj)=ezc1*Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==1
        
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2=ReturnMatrix_z;
            temp2(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^ezc3;
            temp2(ReturnMatrix_z==0)=0; % matlab otherwise puts 0 to negative power to infinity

            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            temp4(isfinite(temp4))=(ezc4*sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV_z==0)=0;
            
            entireRHS_z=ezc2*temp2+DiscountFactorParamsVec*temp4*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z.^ezc7,[],1);
            V(:,z_c,jj)=ezc1*Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            temp4=EV_z;
            temp4(isfinite(temp4))=(ezc4*sj(N_j)*temp4(isfinite(temp4))).^ezc6;
            temp4(EV_z==0)=0;
                  
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                temp2=ReturnMatrix_az;
                temp2(isfinite(ReturnMatrix_az))=ReturnMatrix_az(isfinite(ReturnMatrix_az)).^ezc3;
                temp2(ReturnMatrix_az==0)=0; % matlab otherwise puts 0 to negative power to infinity

                entireRHS_az=ezc2*temp2+DiscountFactorParamsVec*temp4;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az.^ezc7);
                V(a_c,z_c,jj)=ezc1*Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end


end