function [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

%% Check which vfoptions have been used, set all others to defaults 
if nargin<13
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.tolerance=10^(-9);
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=2;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;vfoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        vfoptions.parallel=2;
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    eval('fieldexists=1;vfoptions.lowmemory;','fieldexists=0;')
    if fieldexists==0
        vfoptions.lowmemory=0;
    end
%     eval('fieldexists=1;vfoptions.exoticpreferences;','fieldexists=0;')
%     if fieldexists==0
%         vfoptions.exoticpreferences=0;
%     end  
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
    eval('fieldexists=1;vfoptions.returnmatrix;','fieldexists=0;')
    if fieldexists==0
        if isa(ReturnFn,'function_handle')==1;
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
%     eval('fieldexists=1;vfoptions.tolerance;','fieldexists=0;')
%     if fieldexists==0
%         vfoptions.tolerance=10^(-9);
%     end
    eval('fieldexists=1;vfoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        vfoptions.polindorval=1;
    end
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

%% 

if N_d==0
    [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,N_j]);
Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);

end