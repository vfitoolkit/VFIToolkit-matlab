function [V,Policy2]=ValueFnIter_Case1_FHorz_EpsteinZin_noz_e_raw(n_d,n_a,n_e,N_j, d_grid, a_grid,e_grid,pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end



%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
if length(DiscountFactorParamsVec)>3
    DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
end

if fieldexists_pi_e_J==1
    e_grid=vfoptions.e_grid_J(:,N_j);
    pi_e=vfoptions.pi_e_J(:,N_j);
elseif fieldexists_EiidShockFn==1
    if fieldexists_EiidShockFnParamNames==1
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,N_j);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    else
        [e_grid,pi_e]=vfoptions.ExogShockFn(N_j);
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    end
end

pi_e=shiftdim(pi_e,-2); % Move to third dimension

if vfoptions.lowmemory>0
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a,n_e, d_grid, a_grid,e_grid, ReturnFnParamsVec);
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

        %if vfoptions.returnmatrix==2 % GPU
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            % Note: would raise to 1-1/psi, and then to 1/(1-1/psi). So can just
            % skip this and alter the (1-beta) term appropriately. Further, as this
            % is just multiplying by a constant nor will it effect the argmax, so
            % can just scale solution to the max directly.
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=((1-DiscountFactorParamsVec(1))*Vtemp.^(1/(1-1/DiscountFactorParamsVec(3))));
            Policy(:,e_c,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    VKronNext_j=sum(V_Jplus1.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a,n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
        
        %Calc the expectation term (except beta)
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;

        entireEV=kron(temp,ones(N_d,1));
        temp4=entireEV;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp4(entireEV==0)=0;
        
        entireRHS=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(1)*temp4*ones(1,N_a,N_e);
        % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
        % the whole entireRHS. This will be a monotone function, so just find the max, and
        % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
        Policy(:,:,N_j)=maxindex;
        
    elseif vfoptions.lowmemory==1
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;

        entireEV=kron(temp,ones(N_d,1));
        temp4=entireEV;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp4(entireEV==0)=0;
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2=ReturnMatrix_e;
            temp2(isfinite(ReturnMatrix_e))=ReturnMatrix_e(isfinite(ReturnMatrix_e)).^(1-1/DiscountFactorParamsVec(3));
            
            entireRHS_e=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec*temp4*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,e_c,N_j)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
            Policy(:,e_c,N_j)=maxindex;
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
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    if length(DiscountFactorParamsVec)>3
        DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
    end
    
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,jj);
        pi_e=vfoptions.pi_e_J(:,jj);
        pi_e=shiftdim(pi_e,-2); % Move to thrid dimension
    elseif fieldexists_EiidShockFn==1
        if fieldexists_EiidShockFnParamNames==1
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        else
            [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        end
        pi_e=shiftdim(pi_e,-2); % Move to third dimension
    end
    
    if vfoptions.lowmemory>0
        if (fieldexists_pi_e_J==1 || fieldexists_EiidShockFn==1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end
    end
    
    VKronNext_j=V(:,:,jj+1);

    VKronNext_j=sum(VKronNext_j.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a,n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        temp2=ReturnMatrix;
        temp2(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
        
        %Calc the expectation term (except beta)
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;

        entireEV=kron(temp,ones(N_d,1));
        temp4=entireEV;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp4(entireEV==0)=0;
        
        entireRHS=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec(1)*temp4*ones(1,N_a,N_e);
        % No need to compute the .^(1/(1-1/DiscountFactorParamsVec(3))) of
        % the whole entireRHS. This will be a monotone function, so just find the max, and
        % then compute .^(1/(1-1/DiscountFactorParamsVec(3))) of the max.
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
        Policy(:,:,jj)=maxindex;
        
    elseif vfoptions.lowmemory==1
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        temp=VKronNext_j;
        temp(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
        temp(VKronNext_j==0)=0;

        entireEV=kron(temp,ones(N_d,1));
        temp4=entireEV;
        temp4(isfinite(temp4))=temp4(isfinite(temp4)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2)));
        temp4(entireEV==0)=0;
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            temp2=ReturnMatrix_e;
            temp2(isfinite(ReturnMatrix_e))=ReturnMatrix_e(isfinite(ReturnMatrix_e)).^(1-1/DiscountFactorParamsVec(3));
            
            entireRHS_e=(1-DiscountFactorParamsVec(1))*temp2+DiscountFactorParamsVec*temp4*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,e_c,jj)=Vtemp.^(1/(1-1/DiscountFactorParamsVec(3)));
            Policy(:,e_c,jj)=maxindex;
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end