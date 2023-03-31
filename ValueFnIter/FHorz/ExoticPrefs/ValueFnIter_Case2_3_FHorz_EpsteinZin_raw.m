function [V,Policy]=ValueFnIter_Case2_3_FHorz_EpsteinZin_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DiscountFactorParamNames contains the names for the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [ (1-beta)*u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function. c_t if just consuption, or ((c_t)^v(1-l_t)^(1-v)) if consumption and leisure (1-l_t)
%  psi is the elasticity of intertemporal solution
%  gamma is a measure of risk aversion, bigger gamma is more risk averse
%  beta is the standard marginal rate of time preference (discount factor)
%  When 1/(1-psi)=1-gamma, i.e., we get standard von-Neumann-Morgenstern

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z

%%
if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
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

%%

%% j=N_j

if vfoptions.verbose==1
    sprintf('Age j is currently %i \n',N_j)
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
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


if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    % Modify the Return Function appropriately for Epstein-Zin Preferences
    % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
    % skip this and alter the (1-beta) term appropriately. Further, as this
    % is just multiplying by a constant nor will it effect the argmax, so
    % can just scale solution to the max directly.
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=((1-DiscountFactorParamsVec(1))*Vtemp.^(1/(1-1/DiscountFactorParamsVec(3))));
    Policy(:,:,N_j)=maxindex;
    
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
        % skip this and alter the (1-beta) term appropriately. Further, as this
        % is just multiplying by a constant nor will it effect the argmax, so
        % can just scale solution to the max directly.
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=((1-DiscountFactorParamsVec(1))*Vtemp.^(1/(1-1/DiscountFactorParamsVec(3))));
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
        % skip this and alter the (1-beta) term appropriately. Further, as this
        % is just multiplying by a constant nor will it effect the argmax, so
        % can just scale solution to the max directly.
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a,[],1);
        V(a_c,:,N_j)=((1-DiscountFactorParamsVec(1))*Vtemp.^(1/(1-1/DiscountFactorParamsVec(3))));
        Policy(a_c,:,N_j)=maxindex;
    end
    
end

%%
% Case2_Type==3  % phi_a'(d,z')
if vfoptions.phiaprimedependsonage==0
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
end

for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    
    if vfoptions.verbose==1
        sprintf('Age j is currently %i \n',jj)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    if length(DiscountFactorParamsVec)>3
        DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
    end

    if vfoptions.phiaprimedependsonage==1
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
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
    
    Vnextj=V(:,:,jj+1);
        
    if vfoptions.lowmemory==0
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=Vnextj(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^(1-DiscountFactorParamsVec(2));
        EV=reshape(EV,[N_d,N_z]);
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        ReturnMatrix(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));

        EV=EV.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        EV(isfinite(EV))=EV(isfinite(EV)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2))); % More of the Epstein-Zin preferences

        entireRHS=(1-DiscountFactorParamsVec(1)).*ReturnMatrix+DiscountFactorParamsVec(1)*repmat(EV,1,N_a,1);
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
        % the whole entireRHS. This will be a monotone function, so just find the max, and
        % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.

        V(:,:,jj)=shiftdim(Vtemp,1).^(1/(1-1/DiscountFactorParamsVec(3)));
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=Vnextj(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^(1-DiscountFactorParamsVec(2));
        EV=reshape(EV,[N_d,N_z]);
        
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:); % Value of z (not of z')

            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            ReturnMatrix_z(isfinite(ReturnMatrix_z))=ReturnMatrix_z(isfinite(ReturnMatrix_z)).^(1-1/DiscountFactorParamsVec(3));

            EV_z=EV.*kron(pi_z(z_c,:),ones(N_d,1,'gpuArray'));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,1]);

            EV_z(isfinite(EV_z))=EV_z(isfinite(EV_z)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2))); % More of the Epstein-Zin preferences

            entireRHS=(1-DiscountFactorParamsVec(1)).*ReturnMatrix_z+DiscountFactorParamsVec(1)*EV_z.*ones(1,N_a); %d by a (by z)
            
            %calculate in order, the maximizing aprime indexes
            [Vtemp,Policy(:,z_c,jj)]=max(entireRHS,[],1);
            % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
            % the whole entireRHS. This will be a monotone function, so just find the max, and
            % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.
            V(:,z_c,jj)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
        end
    elseif vfoptions.lowmemory==2
        Phi_aprimeMatrix=reshape(Phi_aprimeMatrix,[N_d*N_z,1]);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d,1));
        EV=Vnextj(Phi_aprimeMatrix+N_a*(zprime_ToMatchPhi-1));
        EV(isfinite(EV))=EV(isfinite(EV)).^(1-DiscountFactorParamsVec(2));
        EV=reshape(EV,[N_d,N_z]);
        
        for z_c=1:N_Z
            EV_z=EV.*kron(pi_z(z_c,:),ones(N_d,1,'gpuArray'));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,1]);

            EV_z(isfinite(EV_z))=EV_z(isfinite(EV_z)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2))); % More of the Epstein-Zin preferences

            for a_c=1:N_a
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                ReturnMatrix_az(isfinite(ReturnMatrix_az))=ReturnMatrix_az(isfinite(ReturnMatrix_az)).^(1-1/DiscountFactorParamsVec(3));

                entireRHS=(1-DiscountFactorParamsVec(1)).*ReturnMatrix_az+DiscountFactorParamsVec(1)*EV_z; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [Vtemp,Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
                % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
                % the whole entireRHS. This will be a monotone function, so just find the max, and
                % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.

                V(a_c,z_c,jj)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
            end
        end
    end
end


end