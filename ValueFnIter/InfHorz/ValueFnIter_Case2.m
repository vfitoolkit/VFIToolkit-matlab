function [V, Policy]=ValueFnIter_Case2(n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.lowmemory=0;
    vfoptions.parallel=1+(gpuDeviceCount>0);
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    vfoptions.phiaprimematrix=2;
    vfoptions.polindorval=1;
    vfoptions.howards=60;
    vfoptions.maxhowards=500;
    vfoptions.exoticpreferences=0;
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.policy_forceintegertype=0;
    vfoptions.piz_strictonrowsaddingtoone=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(vfoptions,'returnmatrix')==0
        if vfoptions.parallel==2
            vfoptions.returnmatrix=2; % On GPU, must use this option
        end
        if isfield(vfoptions,'returnmatrix')==0
            if isa(ReturnFn,'function_handle')==1
                vfoptions.returnmatrix=0;
            else
                vfoptions.returnmatrix=1;
            end
        end
    end
    if isfield(vfoptions,'phiaprimematrix')==0
        vfoptions.phiaprimematrix=2;
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'howards')==0
        vfoptions.howards=60;
    end
    if isfield(vfoptions,'maxhowards')==0
        vfoptions.maxhowards=500;
    end
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences=0;
    end  
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'piz_strictonrowsaddingtoone')==0
        vfoptions.piz_strictonrowsaddingtoone=0;
    end
end

if isfield(vfoptions,'V0')
    V0=reshape(vfoptions.V0,[n_a,n_z]);
else
    if vfoptions.parallel==2
        V0=zeros([n_a,n_z], 'gpuArray');
    else
        V0=zeros([n_a,n_z]);        
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
%% 

if min(min(pi_z))<0
    fprintf('ERROR: Problem with pi_z in ValueFnIter_Case2: min(min(pi_z))<0 \n')
    min(min(pi_z))
    dbstack
    return
elseif vfoptions.piz_strictonrowsaddingtoone==1
    if max(sum(pi_z,2))~=1 || min(sum(pi_z,2))~=1
        fprintf('WARNING: Problem with pi_z in ValueFnIter_Case2: rows do not sum to one \n')
        max(sum(pi_z,2))
        min(sum(pi_z,2))
        dbstack
        return
    end
elseif vfoptions.piz_strictonrowsaddingtoone==0
    if max(abs((sum(pi_z,2))-1)) > 10^(-13)
        fprintf('WARNING: Problem with pi_z in ValueFnIter_Case2: rows do not sum to one \n')
        max(sum(pi_z,2))
        min(sum(pi_z,2))
        dbstack
        return
    end
end


%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
% If no ReturnFnParamNames inputted, then figure it out from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_a+l_z)
        ReturnFnParamNames={temp{l_d+l_a+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
    else
        ReturnFnParamNames={};
    end
% else
%     ReturnFnParamNames=ReturnFnParamNames;
end
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

%%
% Create a vector containing all the return function parameters (in order)
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
if vfoptions.phiaprimematrix~=1
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
end

% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
if vfoptions.parallel==2 
   V0=gpuArray(V0);
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

if vfoptions.exoticpreferences==0
    if length(DiscountFactorParamsVec)~=1
        error('There should only be a single Discount Factor (in DiscountFactorParamNames)')
    end
else
    error('Cannot use exotic preferences in Case2 (if you need this please email roberdkirkby@gmail.com')
end

%%
if vfoptions.lowmemory==0
    
    %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension d-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.
    if vfoptions.returnmatrix==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec, vfoptions.parallel);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
    end
    
    if vfoptions.phiaprimematrix==0
        disp('ERROR: vfoptions.phi_aprimematrix==0 has not yet been implemented')
    elseif vfoptions.phiaprimematrix==1
        Phi_aprimeMatrix=Phi_aprime;
        if vfoptions.parallel==2 % If appropriate, make sure that this is on the gpu
            Phi_aprimeMatrix=gpuArray(Phi_aprimeMatrix);
        end
    elseif vfoptions.phiaprimematrix==2 % GPU
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    end
    
    %% The Value Function Iteration
    V0Kron=reshape(V0,[N_a,N_z]);
    
    if vfoptions.parallel==0 % On CPU
        [VKron, Policy]=ValueFnIter_Case2_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,Phi_aprimeMatrix,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance); 
    elseif vfoptions.parallel==1 % On Parallel CPU
        [VKron, Policy]=ValueFnIter_Case2_Par1_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,Phi_aprimeMatrix,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance);
    elseif vfoptions.parallel==2 % On GPU
        [VKron, Policy]=ValueFnIter_Case2_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,Phi_aprimeMatrix,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance);
    end
    
    %% Sort out Policy
    if vfoptions.polindorval==2
        Policy=PolicyInd2Val_Case2(Policy,n_d,n_a,n_z,d_grid,vfoptions.parallel);
    end
    
elseif vfoptions.lowmemory==1
    %% The Value Function Iteration
    V0Kron=reshape(V0,[N_a,N_z]);
    if vfoptions.parallel==2 % On GPU
        if vfoptions.phiaprimematrix~=1
            [VKron, Policy]=ValueFnIter_Case2_LowMem_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn,Phi_aprime,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance, vfoptions.phiaprimematrix, ReturnFnParamsVec,PhiaprimeParamsVec);
        else
            [VKron, Policy]=ValueFnIter_Case2_LowMem_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn,Phi_aprime,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance, vfoptions.phiaprimematrix, ReturnFnParamsVec);
        end
    else
        fprintf('ERROR: ValueFnIter_Case2 only allows lowmemory==1 with parallel==2 \n')
    end
end

V=reshape(VKron,[n_a,n_z]);
if vfoptions.polindorval==1
    Policy=UnKronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,vfoptions);
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean that Policy is not integer valued. The following corrects this.
if vfoptions.policy_forceintegertype==1
    Policy=round(Policy);
end

end