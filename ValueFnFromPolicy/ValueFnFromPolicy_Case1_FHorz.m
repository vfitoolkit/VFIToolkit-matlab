function V=ValueFnFromPolicy_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% vfoptions is an optional input (not required)

if exist('vfoptions','var')==1
    if isfield(vfoptions,'polindorval')==1
        if vfoptions.polindorval==2
            fprintf('ERROR: ValueFnFromPolicy_Case1_FHorz() does not work with policy fn in form of policy values \n');
        end
    end
    if isfield(vfoptions,'exoticpreferences')==1
        if vfoptions.exoticpreferences>0
            fprintf('ERROR: ValueFnFromPolicy_Case1_FHorz() does not yet work with exotic preferences. Please email robertdkirkby@gmail.com if you want/need this feature and I will implement it. \n');
        end
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if isa(Policy, 'gpuArray')
    Parallel=2;
else
    Parallel=1;
end

%% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
if isfield(vfoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
end
if isfield(vfoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
    vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
end
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')

eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_a_temp=l_a;
l_z=length(n_z);
l_e=0;
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
end
l_z_temp=l_z+l_e;
% Figure it out from ReturnFn
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
    ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

if l_e==0
    %% Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
    PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    
    % The following will also be needed to calculate the expectation of next
    % period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);
    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1
        
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
        
        FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,ReturnFnParamNames,jj));
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid_Case1(ReturnFn, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a,N_z]);
   
        if jj==N_j
            V(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,jj+1).*shiftdim(pi_z',-2),2); % size N_z-by-1
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
    
else % l_e>0
    N_e=prod(vfoptions.n_e);
    
    %% Evaluate Return Fn (this part is essentially copy-paste from the 'EvaluateFnOnAgentDist' commands)
    n_z_temp=[n_z,vfoptions.n_e];
    PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z_temp,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z_temp)),1,1+l_a+l_z_temp+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,n_e,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*N_e*(l_d+l_a),N_j]);
    
    % The following will also be needed to calculate the expectation of next
    % period value fn, evaluated based on the policy.
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,vfoptions.n_e);
    
    %% Calculate the Value Fn by backward iteration
    V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    
    for reverse_j=0:N_j-1
        jj=N_j-reverse_j; % current period, counts backwards from J-1
                
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
        if fieldexists_pi_e_J==1
            pi_e=vfoptions.pi_e_J(:,jj);
            e_grid=vfoptions.e_grid_J(:,jj);
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
                [e_grid,pi_e]=vfoptions.ExogShockFn(jj);
                e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
            end
        end
        
        FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,ReturnFnParamNames,jj));
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid_Case1(ReturnFn, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,vfoptions.n_e,l_d+l_a]),n_d,n_a,[n_z,vfoptions.n_e],a_grid,[z_grid;e_grid],Parallel),[N_a,N_z,N_e]);
        
        if jj==N_j
            V(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e,-2),3); % expectation over iid
            EVnext=EVnext.*shiftdim(pi_z',-1); % size N_z-by-1
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
    V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
end



end