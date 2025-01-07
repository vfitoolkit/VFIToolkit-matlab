function V=ValueFnFromPolicy_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% vfoptions is an optional input (not required)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('vfoptions','var')
    if isfield(vfoptions,'parallel')
        if vfoptions.parallel~=2
            error('ValueFnFromPolicy_Case1_FHorz is only available on GPU')
        end
    end
end


%% Exogenous shock grids

% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
if prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
end
if isfield(vfoptions,'ExogShockFn')
    if isfield(vfoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            pi_z_J(:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    else
        for jj=1:N_j
            [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
            pi_z_J(:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    end
end

% If using e variable, do same for this
N_e=0;
if isfield(vfoptions,'n_e')
    if prod(vfoptions.n_e)==0
        vfoptions=rmfield(vfoptions,'n_e');
    else
        n_e=vfoptions.n_e;
        N_e=prod(n_e);
        if isfield(vfoptions,'e_grid_J')
            error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
        end
        if ~isfield(vfoptions,'e_grid') % && ~isfield(vfoptions,'e_grid_J')
            error('You are using an e (iid) variable, and so need to declare vfoptions.e_grid')
        elseif ~isfield(vfoptions,'pi_e')
            error('You are using an e (iid) variable, and so need to declare vfoptions.pi_e')
        end

        e_gridvals_J=zeros(prod(vfoptions.n_e),length(vfoptions.n_e),'gpuArray');
        pi_e_J=zeros(prod(vfoptions.n_e),prod(vfoptions.n_e),'gpuArray');
        if ndims(vfoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                e_gridvals_J=vfoptions.e_grid;
            end
            pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                e_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_e,vfoptions.e_grid(:,jj),1);
            end
            pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) % joint grid
            e_gridvals_J=vfoptions.e_grid.*ones(1,1,N_j,'gpuArray');
            pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1]) % basic grid
            e_gridvals_J=CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
            pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        end
        if isfield(vfoptions,'ExogShockFn')
            if isfield(vfoptions,'ExogShockFnParamNames')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            else
                for jj=1:N_j
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.ExogShockFn(N_j);
                    pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            end
        end
    end
end


%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
% if isfield(vfoptions,'SemiExoStateFn')
%     l_z=length(vfoptions.n_semiz)+l_z;
% end
l_e=0;
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
end
%Figure out ReturnFnParamNames from ReturnFn
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_aprime+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
    ReturnFnParamNames={temp{l_d+l_aprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
else
    ReturnFnParamNames={};
end
% [l_d,l_aprime,l_a,l_z,l_e]
% ReturnFnParamNames
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later


a_gridvals=CreateGridvals(n_a,a_grid,1);

l_daprime=size(Policy,1);

%%
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, 0,N_j);
    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1
        
        PolicyValuesPermute=permute(PolicyValues(:,:,jj),[2,1]); %[N_a,N_z,l_d+l_a]

        % Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
   
        if jj==N_j
            V(:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=V(:,jj+1);
            
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=PolicyIndexesKron(:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,jj),1);
            end
                        
            aprimez_index=reshape(optaprime,[N_a,1]); % N_a*(z_index-1), but just with lots of kron
            
            EVnextOfPolicy=EVnext(aprimez_index);
            
            V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,1]);
        end
    end
    
    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,N_j]);

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_e,N_j,d_grid,a_grid,vfoptions,1);
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_e,N_j);
    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_e,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1
        
        PolicyValuesPermute=permute(PolicyValues(:,:,:,jj),[2,3,1]); %[N_a,N_z,l_d+l_a]

        % Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_e,a_gridvals,e_gridvals_J(:,:,jj));
   
        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(pi_e_J(:,jj),-1),2); % expectation over iid
            
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=PolicyIndexesKron(:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end
                        
            aprimez_index=reshape(optaprime,[N_a*N_e,1])+N_a*(kron((1:1:N_e)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron
            
            EVnextOfPolicy=EVnext(aprimez_index);
            
            V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_e]);
        end
    end
    
    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_e,N_j]);

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);
    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1
        
        PolicyValuesPermute=permute(PolicyValues(:,:,:,jj),[2,3,1]); %[N_a,N_z,l_d+l_a]

        % Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
   
        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-2),2); % size N_z-by-1
            EVnext(isnan(EVnext))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension
%             EVnext=reshape(EVnext,[N_a,N_z]); % Not necessary as just index into it
            
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=PolicyIndexesKron(:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end
                        
            aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron
            
            EVnextOfPolicy=EVnext(aprimez_index);
            
            V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z]);
        end
    end
    
    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,N_j]);
    
elseif N_z>0 && N_e>0
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,n_e);

    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        PolicyValuesPermute=permute(PolicyValues(:,:,:,jj),[2,3,1]); %[N_a,N_z,l_d+l_a]

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,[n_z,n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
        
        if jj==N_j
            V(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3); % expectation over iid
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1); % size N_z-by-1
            EVnext(isnan(EVnext))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension
%             EVnext=reshape(EVnext,[N_a,N_z]); % Not necessary as just index into it

            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=PolicyIndexesKron(:,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,:,jj),1);
            end
            
            aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron
            
            EVnextOfPolicy=EVnext(aprimez_index);
            
            V(:,:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z,N_e]);
        end
        
    end
    
    %Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,n_e,N_j]);
end



end