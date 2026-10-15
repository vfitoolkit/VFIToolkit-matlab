function Obj=EstimateLifeCycleModel_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions)
% Objective function of EstimateLifeCycleModel_MaximumLikelihood: solves the model at the parameters and evaluates the log-likelihood.
% estimoptions.vectoroutput=0: Obj is minus the log-likelihood (to be minimized), with the penalty for parameters that try to leave
%                              their constraints (as in CalibrateLifeCycleModel_objectivefn)
% estimoptions.vectoroutput=1: Obj is LogLi, the N-by-1 vector of the individual log-likelihoods (used for the scores)

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[estimparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(estimparamsvec,estimparamsvecindex,EstimParamNames,estimoptions);

if estimoptions.verbose==1
    fprintf(' \n')
    fprintf('Current parameter values: \n')
    for pp=1:length(EstimParamNames)
        if estimparamsvecindex(pp+1)-estimparamsvecindex(pp)==1
            fprintf(['    ',EstimParamNames{pp},'= %8.6f \n'],estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))
        else
            fprintf(['    ',EstimParamNames{pp},'=  \n'])
            estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))' % want the output as a row
        end
    end
end

for pp=1:length(EstimParamNames)
    if estimomitparams_counter(pp)>0
        currparamraw=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        Parameters.(EstimParamNames{pp})=currparamraw;
    else
        Parameters.(EstimParamNames{pp})=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being estimated (otherwise they are already done)
if estimoptions.calibrateshocks==1
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_gridvals_J,pi_z_J,N_j,Parameters,vfoptions,3,simoptions.jequaloneDist_usergrids);
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
    if simoptions.jequaloneDist_usergrids==1
        simoptions.user_z_grid=vfoptions.user_z_grid;
        simoptions.user_pi_z=vfoptions.user_pi_z;
    end
end

%% Solve the model, and evaluate the log-likelihood
[~, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

[LogL,LogLi]=LogLikelihood_FHorz(Data,MeasurementError,jequaloneDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,simoptions,estimoptions);

%% The objective
if estimoptions.vectoroutput==1
    Obj=LogLi;
else
    Obj=-LogL; % minimize minus the log-likelihood
    if isnan(Obj)
        Obj=Inf;
    end
    if penalty>0
        if Obj>0
            Obj=1.2*penalty*Obj; % 20% penalty for being too far in violation of restrictions
        else % Obj is negative, so penalty is to reduce magnitude
            Obj=0.8*(1/penalty)*Obj; % 20% penalty for being too far in violation of restrictions
        end
    end
end

%% Verbose
if estimoptions.verbose==1
    fprintf('Current log-likelihood is %8.6f \n', LogL)
    if penalty>0 && estimoptions.vectoroutput==0
        if -LogL>0
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 1.2*penalty)
        else
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 0.8*(1/penalty) )
        end
    end
end

end
