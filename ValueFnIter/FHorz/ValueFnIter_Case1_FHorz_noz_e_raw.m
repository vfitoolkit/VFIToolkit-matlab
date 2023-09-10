function [V,Policy2]=ValueFnIter_Case1_FHorz_noz_e_raw(n_d,n_a,n_e,N_j, d_grid, a_grid, e_grid, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
e_grid=gpuArray(e_grid);

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

pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)

if vfoptions.lowmemory>0
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(:,e_c,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e,2);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);  % Because no z, can treat e like z and call Par2 rather than Par2e
        % (d,aprime,a,e)
        
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repmat(V_Jplus1,1,N_a,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            % (d,aprime,a)
            
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*V_Jplus1.*ones(1,N_a);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,e_c,N_j)=shiftdim(maxindex,1);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

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
        pi_e=shiftdim(pi_e,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)
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
        
    VKronNext_j=sum(VKronNext_j.*pi_e,2);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_e, d_grid, a_grid, e_grid, ReturnFnParamsVec);
        % (d,aprime,a,e)
        
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*repmat(VKronNext_j,1,N_a,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_e, d_grid, a_grid, e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*VKronNext_j.*ones(1,N_a);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
     
    end

end

%%
Policy2=zeros(2,N_a,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end