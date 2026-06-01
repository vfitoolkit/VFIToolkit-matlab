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
% V control in Kron form, plus N_ze (combined z*e dim, =1 if neither)
if N_e==0
    if N_z==0
        N_ze=1;
        VcontrolKron=reshape(V.control,[N_a,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_j]);
    else
        N_ze=N_z;
        VcontrolKron=reshape(V.control,[N_a,N_z,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_z,N_j]);
    end
else
    if N_z==0
        N_ze=N_e;
        VcontrolKron=reshape(V.control,[N_a,N_e,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_e,N_j]);
    else
        N_ze=N_z*N_e;
        VcontrolKron=reshape(V.control,[N_a,N_z,N_e,N_j]);
        ValtcontrolKron=reshape(V.control,[N_a,N_z,N_e,N_j]);
    end
end

% Combine Policy.control into uniform 4D [1, N_a, N_ze, N_j] with combined (d + N_d*(aprime-1)) index,
% matching raw VFI output and what UnKronPolicyIndexes1_FHorz_* expects.
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
PolicyTemp=reshape(Policy.control,[l_d+l_a,N_a,N_ze,N_j]);
% combined aprime in [1, prod(n_a)]
if l_a==1
    combined_a=PolicyTemp(l_d+1,:,:,:);
elseif l_a==2
    combined_a=PolicyTemp(l_d+1,:,:,:)+n_a(1)*(PolicyTemp(l_d+2,:,:,:)-1);
elseif l_a==3
    combined_a=PolicyTemp(l_d+1,:,:,:)+n_a(1)*(PolicyTemp(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyTemp(l_d+3,:,:,:)-1);
elseif l_a==4
    combined_a=PolicyTemp(l_d+1,:,:,:)+n_a(1)*(PolicyTemp(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyTemp(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyTemp(l_d+4,:,:,:)-1);
end
if N_d==0
    PolicycontrolKron=reshape(combined_a,[1,N_a,N_ze,N_j]);
else
    % combined d in [1, prod(n_d)]
    if l_d==1
        combined_d=PolicyTemp(1,:,:,:);
    elseif l_d==2
        combined_d=PolicyTemp(1,:,:,:)+n_d(1)*(PolicyTemp(2,:,:,:)-1);
    elseif l_d==3
        combined_d=PolicyTemp(1,:,:,:)+n_d(1)*(PolicyTemp(2,:,:,:)-1)+n_d(1)*n_d(2)*(PolicyTemp(3,:,:,:)-1);
    elseif l_d==4
        combined_d=PolicyTemp(1,:,:,:)+n_d(1)*(PolicyTemp(2,:,:,:)-1)+n_d(1)*n_d(2)*(PolicyTemp(3,:,:,:)-1)+n_d(1)*n_d(2)*n_d(3)*(PolicyTemp(4,:,:,:)-1);
    end
    PolicycontrolKron=reshape(combined_d+N_d*(combined_a-1),[1,N_a,N_ze,N_j]);
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

    % Align PolicyKron_jp with PolicycontrolKron 4D shape [1, N_a, N_ze, TreatmentDuration]
    if N_d==0
        PolicyKron_jp=reshape(PolicyKron_jp,[1,size(PolicyKron_jp)]);
    end
    PolicyKron_jp=reshape(PolicyKron_jp,[1,N_a,N_ze,TreatmentDuration]);

    % Combine VKron_jp and PolicyKron_jp with the control versions
    VKron=VcontrolKron;
    PolicyKron=PolicycontrolKron;
    % VKron rank varies per case; PolicyKron is uniformly 4D
    if N_e==0
        if N_z==0
            VKron(:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
        else
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
        end
    else
        if N_z==0
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
        else
            VKron(:,:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
        end
    end
    PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;

    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    % Reshape PolicyKron from uniform 4D back to per-case shape UnKron expects
    if N_z==0 && N_e==0
        PolicyKron=reshape(PolicyKron,[1,N_a,N_j]);
    elseif N_z>0 && N_e>0
        PolicyKron=reshape(PolicyKron,[1,N_a,N_z,N_e,N_j]);
    end
    if N_e==0
        if N_z==0
            V_jp=reshape(VKron,[n_a,N_j]);
            if N_d==0
                Policy_jp=UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_a, n_a, N_j, vfoptions);
            else
                Policy_jp=UnKronPolicyIndexes1_FHorz_noz(PolicyKron, [n_d,n_a], n_a, N_j, vfoptions);
            end
        else
            V_jp=reshape(VKron,[n_a,n_z,N_j]);
            if N_d==0
                Policy_jp=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_a, n_a, n_z, N_j, vfoptions);
            else
                Policy_jp=UnKronPolicyIndexes1_FHorz_z(PolicyKron, [n_d,n_a], n_a, n_z, N_j, vfoptions);
            end
        end
    else
        if N_z==0
            V_jp=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            % Treat e as z (because no z)
            if N_d==0
                Policy_jp=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_a, n_a, vfoptions.n_e, N_j, vfoptions);
            else
                Policy_jp=UnKronPolicyIndexes1_FHorz_z(PolicyKron, [n_d,n_a], n_a, vfoptions.n_e, N_j, vfoptions);
            end
        else
            V_jp=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if N_d==0
                Policy_jp=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_a, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            else
                Policy_jp=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, [n_d,n_a], n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end

    V.(['treatmentage',num2str(j_p)])=V_jp;
    Policy.(['treatmentage',num2str(j_p)])=Policy_jp;


end


end
