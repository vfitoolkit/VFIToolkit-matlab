function [V, Policy]=ValueFnIter_FieldExp_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TreatmentParams, TreatmentAgeRange, TreatmentDuration, vfoptions)

V=struct();
Policy=struct();

if isempty(n_d)
    n_d=0;
end
if isempty(n_z)
    n_z=0;
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% First, just solve the control group
[V.control, Policy.control, Valt.control]=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% I want the Kron versions of V.control as it makes combining the treatment and control easier (which is done below)
if N_e==0
    if N_z==0
        VcontrolKron=reshape(V.control,[N_a,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1_noz(Policy.control, n_d, n_a, N_j);
    else
        VcontrolKron=reshape(V.control,[N_a,N_z,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_z,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, n_z, N_j);
    end
else
    if N_z==0
        VcontrolKron=reshape(V.control,[N_a,N_e,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_e,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, vfoptions.n_e, N_j); % Treat e as z (because no z)
    else
        VcontrolKron=reshape(V.control,[N_a,N_z,N_e,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_z,N_e,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, n_z, N_j, vfoptions.n_e);
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
if N_z==0
    l_z=0;
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end
% If no ReturnFnParamNames inputted, then figure it out from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
end

%% 
% make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

if vfoptions.verbose==1
    vfoptions
end


%% Change the parameters to treatment parameters
treatparamnames=fieldnames(TreatmentParams);
for nn=1:length(treatparamnames)
    Parameters.(treatparamnames{nn})=TreatmentParams.(treatparamnames{nn});
end

% Identify the age-dependent parameters so that we can easily use just the
% parts relevant to the treatment ages
AgeDepParams=struct();
paramnames=fieldnames(Parameters);
for nn=1:length(paramnames)
    if length(Parameters.(paramnames{nn}))==N_j
        AgeDepParams.(paramnames{nn})=Parameters.(paramnames{nn});
    end
end
agedepparamnames=fieldnames(AgeDepParams);

vfoptions.outputkron=1;
%% Now, loop over the ages in 
for j_p=TreatmentAgeRange(1):TreatmentAgeRange(2)
    % First, get the 'V in period j_p+j_d' from V.control
    vfoptions.V_Jplus1=Valt.control(j_p+TreatmentDuration);
    
    % Instead of N_j we are going to use TreatmentDuration

    % We need to replace the age-dependent parameters with just the relevant periods.
    for nn=1:length(agedepparamnames)
        temp=AgeDepParams.(agedepparamnames{nn});
        Parameters.(agedepparamnames{nn})=temp(j_p:j_p+TreatmentDuration-1);
    end

    %% Solve value fn problem for treatment periods
    % Note: TreatmentDuration periods
    % Note: vfoptions.outputkron=1;
    [VKron_jp, PolicyKron_jp]=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,TreatmentDuration,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

    % Combine VKron_jp and PolicyKron_jp with the control versions
    VKron=VcontrolKron;
    PolicyKron=PolicycontrolKron;
    if N_e==0
        if N_z==0
            VKron(:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        else
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        end
    else
        if N_z==0
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        else
            VKron(:,:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        end
    end
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_e==0
        if N_z==0
            V_jp=reshape(VKron,[n_a,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V_jp=reshape(VKron,[n_a,n_z,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    else
        if N_z==0
            V_jp=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V_jp=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end

    V.(['treatmentage',num2str(j_p)])=V_jp;
    Policy.(['treatmentage',num2str(j_p)])=Policy_jp;


end


end
