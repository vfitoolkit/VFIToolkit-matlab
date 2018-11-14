function [V, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)


%% Check which vfoptions have been used, set all others to defaults 
if ~exist('vfoptions','var')
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.tolerance=10^(-9);
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=2;
    vfoptions.returnmatrix=2;
    vfoptions.phiaprimematrix=2;
    vfoptions.phiaprimedependsonage=0;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.nphi=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.Dynasty_CareAboutDecendents=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')
        vfoptions.parallel=2;
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'phiaprimematrix')
        vfoptions.phiaprimematrix=2;
    end
    if isfield(vfoptions,'phiaprimedependsonage')
        vfoptions.phiaprimedependsonage=0;
    end
    if isfield(vfoptions,'nphi')
        vfoptions.nphi=1;
    end
    if isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
%     if isfield(vfoptions,'exoticpreferences')
%         vfoptions.exoticpreferences=0;
%     end  
    if isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'returnmatrix')
        if isa(ReturnFn,'function_handle')==1;
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
%     if isfield(vfoptions,'tolerance')
%         vfoptions.tolerance=10^(-9);
%     end
    if isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'Dynasty_CareAboutDecendents')==0
        vfoptions.Dynasty_CareAboutDecendents=0;
    end
end

% Check for age dependent grids
if isfield(vfoptions,'agedependentgrids')
    % Some of the grid sizes vary by age, so send to the relevant subcommand
    [V, Policy]=ValueFnIter_Case2_FHorz_AgeDependentGrids(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    return
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check the sizes of some of the inputs
if size(d_grid)~=[N_d, 1]
    disp('ERROR: d_grid is not the correct shape (should be  of size N_d-by-1)')
    dbstack
    return
elseif size(a_grid)~=[N_a, 1]
    disp('ERROR: a_grid is not the correct shape (should be  of size N_a-by-1)')
    dbstack
    return
elseif size(z_grid)~=[N_z, 1]
    disp('ERROR: z_grid is not the correct shape (should be  of size N_z-by-1)')
    dbstack
    return
elseif size(pi_z)~=[N_z, N_z]
    disp('ERROR: pi is not of size N_z-by-N_z')
    dbstack
    return
end

%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
% else
%    % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
%    % This may be completely unnecessary.
%    pi_z=gather(pi_z);
%    d_grid=gather(d_grid);
%    a_grid=gather(a_grid);
%    z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

% if vfoptions.exoticpreferences==0
%     if length(DiscountFactorParamNames)~=1
%         disp('WARNING: There should only be a single Discount Factor (in DiscountFactorParamNames) when using standard VFI')
%         dbstack
%     end
% elseif vfoptions.exoticpreferences==1 % Multiple discount factors. It is assumed that the product
%     %NOT YET IMPLEMENTED
% %    [V, Policy]=ValueFnIter_Case1_QuasiGeometric(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
% %    return
% elseif vfoptions.exoticpreferences==2 % Epstein-Zin preferences
%     %NOT YET IMPLEMENTED
% %     [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
% %     return
% end

%% Deal with Dynasty_CareAboutDecendents if need to do that.
if vfoptions.Dynasty_CareAboutDecendents==0
    if vfoptions.verbose==1
        fprintf('Dynasty_CareAboutDecendents option is being used \n')
    end
    if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    
    if vfoptions.nphi==1
        if vfoptions.parallel==0
            disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_Dynasty_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        elseif vfoptions.parallel==2
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_Dynasty_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        end
    else
        if vfoptions.parallel==0
            disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Dynasty_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        elseif vfoptions.parallel==2
            [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Dynasty_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        end
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
    
end

%% 
if vfoptions.nphi==1
    if vfoptions.parallel==0
        disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
        [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    elseif vfoptions.parallel==2
        [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    end
else
    if vfoptions.parallel==0
        disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
        [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    elseif vfoptions.parallel==2
        [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_nphi_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    end
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

end