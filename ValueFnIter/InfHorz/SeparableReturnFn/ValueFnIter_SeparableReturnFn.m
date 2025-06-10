function [V,Policy]=ValueFnIter_SeparableReturnFn(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

ReturnFnParamNames.R1=ReturnFnParamNamesFn(ReturnFn.R1,n_d,n_a,n_z,0,vfoptions,Parameters);
ReturnFnParamNames.R2=ReturnFnParamNamesFn(ReturnFn.R2,n_d,n_a,n_z,0,vfoptions,Parameters);

% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
V0=gpuArray(V0);
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

if vfoptions.verbose==1
    vfoptions
end

%% Switch to z_gridvals
if vfoptions.alreadygridvals==0
    if vfoptions.parallel<2
        % only basics allowed with cpu
        z_gridvals=z_grid;
    else
        [z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);
    end
elseif vfoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec.R1=CreateVectorFromParams(Parameters, ReturnFnParamNames.R1);
ReturnFnParamsVec.R2=CreateVectorFromParams(Parameters, ReturnFnParamNames.R2);

if isfield(vfoptions,'exoticpreferences')
    if vfoptions.exoticpreferences~=3
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
        if vfoptions.exoticpreferences==0
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
end

%% Solve
if strcmp(vfoptions.solnmethod,'purediscretization')
    % TO BE IMPLEMENTED
end

if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
    % TO BE IMPLEMENTED
end

end