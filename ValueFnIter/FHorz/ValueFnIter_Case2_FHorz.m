function [V, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)


%% Check which vfoptions have been used, set all others to defaults 
if ~exist('vfoptions','var')
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
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
    vfoptions.phiaprimedependsonage=0;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.nphi=0;
    vfoptions.policy_forceintegertype=0;
%     vfoptions.exoticpreferences % default is not to declare it
    vfoptions.dynasty=0;
    vfoptions.agedependentgrids=0;
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
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
    if ~isfield(vfoptions,'phiaprimematrix')
        vfoptions.phiaprimematrix=2;
    end
    if ~isfield(vfoptions,'phiaprimedependsonage')
        vfoptions.phiaprimedependsonage=0;
    end
    if ~isfield(vfoptions,'nphi')
        vfoptions.nphi=0;
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
%     vfoptions.exoticpreferences % default is not to declare it
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    elseif vfoptions.dynasty==1
        if ~isfield(vfoptions,'dynasty_howards')
            vfoptions.dynasty_howards=10;
        end
        if ~isfield(vfoptions,'dynasty_maxhowards')
            vfoptions.dynasty_maxhowards=100;
        end
    end
    if ~isfield(vfoptions,'agedependentgrids')
        vfoptions.agedependentgrids=0;
    end
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
end

% Check for age dependent grids
if prod(vfoptions.agedependentgrids)~=0
    % Some of the grid sizes vary by age, so send to the relevant subcommand
    [V, Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    return
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check the sizes of some of the inputs
if ~all(size(d_grid)==[sum(n_d), 1])
    if ~isempty(n_d) % Make sure d is being used before complaining about size of d_grid
        if n_d~=0
            error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        end
    end
elseif ~all(size(a_grid)==[sum(n_a), 1])
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1)')
elseif ~all(size(z_grid)==[sum(n_z), 1]) && ~all(size(z_grid)==[prod(n_z),length(n_z)])
    if N_z>0
        error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
    end
elseif ~isequal(size(pi_z), [N_z, N_z])
    if N_z>0
        error('pi is not of size N_z-by-N_z')
    end
elseif isfield(vfoptions,'n_e')
    if ~isfield(vfoptions,'e_grid') && ~isfield(vfoptions,'e_grid_J')
        error('When using vfoptions.n_e you must declare vfoptions.e_grid (or vfoptions.e_grid_J)')
    elseif ~isfield(vfoptions,'pi_e') && ~isfield(vfoptions,'pi_e_J')
        error('When using vfoptions.n_e you must declare vfoptions.pi_e (or vfoptions.pi_e_J)')
    else
        % check size of e_grid and pi_e
        if isfield(vfoptions,'e_grid')
            if  ~all(size(vfoptions.e_grid)==[sum(vfoptions.n_e), 1]) && ~all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)])
                error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1)')
            end
        else % using e_grid_J
            % HAVE NOT YET IMPLEMENTED A CHECK OF THE SIZE OF e_grid_J
        end
        if isfield(vfoptions,'pi_e')
            if ~all(size(vfoptions.pi_e)==[prod(vfoptions.n_e),1])
                error('vfoptions.pi_e is not the correct shape (should be of size N_e-by-1)')
            end
        else % using pi_e_J
            if ~all(size(vfoptions.pi_e_J)==[prod(vfoptions.n_e),N_j])
                error('vfoptions.pi_e_J is not the correct shape (should be of size N_e-by-N_j)')
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
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
else
    l_e=0;
end
% Figure out ReturnFnParamNames from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,a,z) for Case2
    else
        ReturnFnParamNames={};
    end
end
% Figure out PhiaprimeParamNames from Phi_aprime
if isempty(PhiaprimeParamNames)
    temp=getAnonymousFnInputNames(Phi_aprime);

    % NOTE: FOLLOWING LARGELY OMITS POSSIBILITY OF e VARIABLES
    if Case2_Type==1 % phi_a'(d,a,z,z')
        if length(temp)>(l_d+l_a+l_z+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+l_z+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==11 || Case2_Type==12 % phi_a'(d,a,z') OR phi_a'(d,a,z)
        if length(temp)>(l_d+l_a+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==2  % phi_a'(d,z,z')
        if length(temp)>(l_d+l_z+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_z+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==3  % phi_a'(d,z')
        if length(temp)>(l_d+l_z) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_z+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    elseif Case2_Type==4  % phi_a'(d,a)
        if length(temp)>(l_d+l_a) % This is largely pointless, the Phi_aprime is always going to have some parameters
            PhiaprimeParamNames={temp{l_d+l_a+1:end}}; % the first inputs will always be (d,a,z) for Case2
        else
            PhiaprimeParamNames={};
        end
    end
end

%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   d_grid=gather(d_grid);
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

%% Exotic Preferences
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_Case2_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn,Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        return
    end
end


%% Deal with Dynasty_CareAboutDecendents if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('Dynasty_CareAboutDecendents option is being used \n')
    end
    if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    
    if vfoptions.nphi==0
        if vfoptions.parallel==0
            disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_Dynasty_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        elseif vfoptions.parallel==2
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_Dynasty_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        end
    elseif vfoptions.nphi==1
        disp('WARNING: FINITE HORZ VALUEFNITER CODES DONT YET WORK WITH nphi \n')
        dbstack
%         if vfoptions.parallel==0
%             disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2) \n')
%             [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Dynasty_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
%         elseif vfoptions.parallel==2
%             [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Dynasty_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
%         end
    end
    
    % Transform V & PolicyIndexes out of kroneckered form
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z,N_j,vfoptions);
    
    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        Policy=uint64(Policy);
        Policy=double(Policy);
    end
    
    return
end

%%
N_e=0;
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
end

if vfoptions.nphi==0
    if vfoptions.parallel==0
        error('Case2 in FHorz only works on gpu (parallel=2)')
%         [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    elseif vfoptions.parallel==2
        if N_e>0
            if Case2_Type==3  % phi_a'(d,z')
                [VKron,PolicyKron]=ValueFnIter_Case2_3_FHorz_e_Par2_raw(n_d,n_a,n_z,vfoptions.n_e, N_j, d_grid, a_grid, z_grid, vfoptions.e_grid, pi_z, vfoptions.pi_e,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            else
                error('e variables are only supported for Case2_Type=3, if you need some other Case2_Type please contact me')
            end
        else
            if Case2_Type==1 % phi_a'(d,a,z,z')
                [VKron,PolicyKron]=ValueFnIter_Case2_1_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            elseif Case2_Type==11 % phi_a'(d,a,z')
                [VKron,PolicyKron]=ValueFnIter_Case2_11_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            elseif Case2_Type==12 % phi_a'(d,a,z)
                [VKron,PolicyKron]=ValueFnIter_Case2_12_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            elseif Case2_Type==2  % phi_a'(d,z,z')
                [VKron,PolicyKron]=ValueFnIter_Case2_2_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            elseif Case2_Type==3  % phi_a'(d,z')
                [VKron,PolicyKron]=ValueFnIter_Case2_3_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            elseif Case2_Type==4  % phi_a'(d,a)
                [VKron,PolicyKron]=ValueFnIter_Case2_4_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
            end
        end
    end
else
    error('Case2 in FHorz does not yet work with nphi')
%     if vfoptions.parallel==0
%         disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
%         [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
%     elseif vfoptions.parallel==2
%         [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
%     end
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if N_e==0
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z,vfoptions.n_e,N_j,vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end

%% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end

end