function [V, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon 'Case 1' value function problems with mixture of Endogenous Exit and Exogenous Exit.

% It is only intended that this is called indirectly by ValueFnIter_Case1()

% if ~isfield(vfoptions,'endogenousexit')
%     vfoptions.endogenousexit=0;
% end
% if ~isfield(vfoptions,'endofperiodexit') % THIS IS ANYWAY BEING DONE BY vfoptions.endogenousexit=2
%     vfoptions.endofperiodexit=0; % This has not yet been implemented as an option that can be activated.
% end
if ~isfield(vfoptions,'keeppolicyonexit') % This is ignored by vfoptions.endogenousexit=2 as it hard-codes default value since this is required.
    vfoptions.keeppolicyonexit=0;
end

V=nan; % Matlab was complaining that V was not assigned

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
d_gridvals=CreateGridvals(n_d,d_grid,1);

% Make sure that the inputs specifically required for mix of endogenous exit and exogenous exit have been included.
if ~isfield(vfoptions,'exitprobabilities')
    fprintf('ERROR: vfoptions.endogenousexit=2 requires that you specify vfoptions.exitprobabilities \n');
    return
end
if ~isfield(vfoptions,'endogenousexitcontinuationcost')
    fprintf('ERROR: vfoptions.endogenousexit=2 requires that you specify vfoptions.endogenousexitcontinuationcost \n');
    return
end
if ~isfield(vfoptions,'ReturnToExitFn')
    fprintf('ERROR: vfoptions.endogenousexit=2 requires that you specify vfoptions.ReturnToExitFn \n');
    return
end

%%
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
% The 'return to exit function' parameters (in order)
% The 'return to exit function' parameters (in order)
temp=getAnonymousFnInputNames(vfoptions.ReturnToExitFn);
if length(temp)>(l_a+l_z)
    ReturnToExitFnParamNames={temp{l_a+l_z+1:end}}; % the first inputs will always be (a,z)
else
    ReturnToExitFnParamNames={};
end
ReturnToExitFnParamsVec=CreateVectorFromParams(Parameters, ReturnToExitFnParamNames);
% Parameters relating to 'mixed' exit.
exitprobabilities=CreateVectorFromParams(Parameters, vfoptions.exitprobabilities);
exitprobabilities=[1-sum(exitprobabilities),exitprobabilities];
endogenousexitcontinuationcost=CreateVectorFromParams(Parameters, vfoptions.endogenousexitcontinuationcost);


%%
if vfoptions.lowmemory==0

    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    % Because exit is not until the end of period the return to exit is allowed to depend on aprime, and d.
    if vfoptions.returnmatrix==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_CPU(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        ReturnToExitMatrix=CreateReturnFnMatrix_Disc_CPU(vfoptions.ReturnToExitFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnToExitFnParamsVec);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
        ReturnToExitMatrix=vfoptions.ReturnToExitFn; % It is simply assumed that you are doing this for both.
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, ReturnFnParamsVec);
        ReturnToExitMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(vfoptions.ReturnToExitFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, ReturnToExitFnParamsVec);
    end

    %%
    V0Kron=reshape(V0,[N_a,N_z]);

    if n_d(1)==0
%         if vfoptions.parallel==0     % On CPU
%             [VKron,Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_nod_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit, vfoptions.exitprobabilities, vfoptions.endogenousexitcontinuationcost);
%         elseif vfoptions.parallel==1 % On Parallel CPU
%             [VKron,Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_nod_Par1_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit, vfoptions.exitprobabilities, vfoptions.endogenousexitcontinuationcost);
        if vfoptions.parallel==2 % On GPU
            [VKron,Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_nod_Par2_raw(V0Kron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit, exitprobabilities, endogenousexitcontinuationcost); %  a_grid, z_grid,
        end
    else
%         if vfoptions.parallel==0 % On CPU
%             [VKron, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit, vfoptions.exitprobabilities, vfoptions.endogenousexitcontinuationcost);
%         elseif vfoptions.parallel==1 % On Parallel CPU
%             [VKron, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_Par1_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit, vfoptions.exitprobabilities, vfoptions.endogenousexitcontinuationcost);
%         elseif vfoptions.parallel==2 % On GPU
        if vfoptions.parallel==2 % On GPU
            [VKron, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit, exitprobabilities, endogenousexitcontinuationcost);
        end
    end


%% Cleaning up the output
V=reshape(VKron,[n_a,n_z]);
ExitPolicy=reshape(ExitPolicy,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
PolicyWhenExit=UnKronPolicyIndexes_Case1(PolicyWhenExit, n_d, n_a, n_z,vfoptions);

end
