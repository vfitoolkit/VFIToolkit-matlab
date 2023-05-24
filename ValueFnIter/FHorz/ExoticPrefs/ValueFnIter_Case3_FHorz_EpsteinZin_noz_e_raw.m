function [V,Policy]=ValueFnIter_Case3_FHorz_EpsteinZin_noz_e_raw(n_d,n_a,n_e,n_u, N_j, d_grid, a_grid, e_grid, u_grid, pi_e, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
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
N_u=prod(n_u);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
e_grid=gpuArray(e_grid);
u_grid=gpuArray(u_grid);

if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
if length(DiscountFactorParamsVec)>3
    DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a,n_e, d_grid, a_grid,e_grid, ReturnFnParamsVec,0);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,N_j)=((1-DiscountFactorParamsVec(1)).^(1/(1-1/DiscountFactorParamsVec(3))))*Vtemp;
        Policy(:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals(a_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec,0);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=((1-DiscountFactorParamsVec(1)).^(1/(1-1/DiscountFactorParamsVec(3))))*Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
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
    
    if vfoptions.lowmemory>0
        if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
            e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
            e_gridvals=e_grid;
        end
    end

    % Using V_Jplus1
    V_Jplus1=vfoptions.V_Jplus1;
    % Part of Epstein-Zin is before taking expectation
    V_Jplus1(isfinite(V_Jplus1))=V_Jplus1(isfinite(V_Jplus1)).^(1-DiscountFactorParamsVec(2));

    V_Jplus1=sum(reshape(V_Jplus1,[N_a,N_e]).*pi_e',2);    % First, switch V_Jplus1 into Kron form ,take expecation over e

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
        
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*V_Jplus1(aprimeIndex); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*V_Jplus1(aprimeIndex+1); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((aprimeProbs.*EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    % Part of Epstein-Zin is after taking expectation
    EV(isfinite(EV))=EV(isfinite(EV)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2))); % More of the Epstein-Zin preferences

    
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a,n_e, d_grid, a_grid,e_grid, ReturnFnParamsVec);
        % (d,a,e)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        ReturnMatrix(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));
  
        entireRHS=(1-DiscountFactorParamsVec(1))*ReturnMatrix+DiscountFactorParamsVec(1)*EV.*ones(1,N_a,N_e); % d-by-a-by-e
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
       for e_c=1:N_e
           e_val=e_gridvals(e_c,:);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_e, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
           
           % Modify the Return Function appropriately for Epstein-Zin Preferences
           ReturnMatrix_e(isfinite(ReturnMatrix_e))=ReturnMatrix_e(isfinite(ReturnMatrix_e)).^(1-1/DiscountFactorParamsVec(3));
           
           entireRHS_e=(1-DiscountFactorParamsVec(1))*ReturnMatrix_e+DiscountFactorParamsVec(1)*EV; % d-by-a
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
           V(:,e_c,N_j)=Vtemp;
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
    
    VKronNext_j=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation
    VKronNext_j(isfinite(VKronNext_j))=VKronNext_j(isfinite(VKronNext_j)).^(1-DiscountFactorParamsVec(2));
    
    VKronNext_j=sum(VKronNext_j.*pi_e',2); % Expectation over e
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*VKronNext_j(aprimeIndex); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*VKronNext_j(aprimeIndex+1); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((aprimeProbs.*EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    % Part of Epstein-Zin is after taking expectation
    EV(isfinite(EV))=EV(isfinite(EV)).^((1-1/DiscountFactorParamsVec(3))/(1-DiscountFactorParamsVec(2))); % More of the Epstein-Zin preferences

    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);
        % (d,a,e)
        
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        ReturnMatrix(isfinite(ReturnMatrix))=ReturnMatrix(isfinite(ReturnMatrix)).^(1-1/DiscountFactorParamsVec(3));

        entireRHS=(1-DiscountFactorParamsVec(1))*ReturnMatrix+DiscountFactorParamsVec(1)*EV.*ones(1,N_a,N_e); % d-by-a-by-e
        
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
       
       betaEV=DiscountFactorParamsVec(1)*EV.*ones(1,N_a,1);
        
       for e_c=1:N_e
           e_val=e_gridvals(e_c,:);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
           
           % Modify the Return Function appropriately for Epstein-Zin Preferences
           ReturnMatrix_e(isfinite(ReturnMatrix_e))=ReturnMatrix_e(isfinite(ReturnMatrix_e)).^(1-1/DiscountFactorParamsVec(3));

           entireRHS_e=(1-DiscountFactorParamsVec(1))*ReturnMatrix_e+betaEV; % d-by-a; % d-by-a
           
           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e.^(1/(1-1/DiscountFactorParamsVec(3))),[],1);
           V(:,e_c,jj)=Vtemp;
           Policy(:,e_c,jj)=maxindex;
        end
        
    end
end


end