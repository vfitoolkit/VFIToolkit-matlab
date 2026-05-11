function V=ValueFnFromPolicy_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% vfoptions is an optional input (not required)
% we will fill in defaults if needed (e.g. `gridinterplayer`).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('vfoptions','var')
    if isfield(vfoptions,'parallel')
        if vfoptions.parallel~=2
            error('ValueFnFromPolicy_FHorz is only available on GPU')
        end
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
else
    vfoptions=struct();
    vfoptions.gridinterplayer=0;
end


%% Exogenous shock grids
% Switch to z_gridvals
[z_gridvals_J, pi_z_J,vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
% Convert z and e to age-dependent joint-grids and transtion matrix
% output: z_gridvals_J, pi_z_J, options.e_gridvals_J, options.pi_e_J

if isfield(vfoptions,'n_semiz')
    error('cannot yet handle semiz, ask on forum if you need/want')
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
if N_z==0
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
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
   
        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % expectation over iid
            
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
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
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
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1_e(Policy, n_d, n_a, n_z, n_e, N_j, vfoptions);

    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1

        PolicyValuesPermute=permute(PolicyValues(:,:,:,jj),[2,3,1]); %[N_a,N_z,l_d+l_a]

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,[n_z,n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
        
        if jj==N_j
            V(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % expectation over iid
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1); % size N_z-by-1
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
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
    
    % Transforming Value Fn out of Kronecker Form
    V=reshape(V,[n_a,n_z,n_e,N_j]);
end



end
