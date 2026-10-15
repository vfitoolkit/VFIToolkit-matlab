function Obj=EstimateLifeCycleModel_PType_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions)
% Objective function of EstimateLifeCycleModel_PType_MaximumLikelihood: solves the model (every permanent type) at the parameters and
% evaluates the log-likelihood (the finite mixture over the types, LogLikelihood_FHorz_PType).
% estimoptions.vectoroutput=0: Obj is minus the log-likelihood (to be minimized), with the penalty for parameters that try to leave
%                              their constraints (as in CalibrateLifeCycleModel_PType_objectivefn)
% estimoptions.vectoroutput=1: Obj is LogLi, the N-by-1 vector of the individual log-likelihoods (used for the scores)
% estimoptions.vectoroutput=2: Obj is [LogLi, PTypePosterior], N-by-(1+N_i)

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[estimparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(estimparamsvec,estimparamsvecindex,EstimParamNames,estimoptions);
% Note: ptype makes no difference to this (the constraints are by block, a block being a parameter, or a parameter of one type).

if estimoptions.verbose==1
    fprintf(' \n')
    fprintf('Current parameter values: \n')
    for pp=1:nEstimParams
        if nEstimParamsFinder(pp,2)==0 % parameter does not depend on ptype
            tempname=EstimParamNames{nEstimParamsFinder(pp,1)};
        else
            tempname=[EstimParamNames{nEstimParamsFinder(pp,1)},'.',Names_i{nEstimParamsFinder(pp,2)}];
        end
        if estimparamsvecindex(pp+1)-estimparamsvecindex(pp)==1
            fprintf(['    ',tempname,'= %8.6f \n'],estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))
        else
            fprintf(['    ',tempname,'=  \n'])
            estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))' % want the output as a row
        end
    end
end

for pp=1:nEstimParams
    if estimomitparams_counter(pp)>0
        currparamraw=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
    else
        currparamraw=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
    end
    if nEstimParamsFinder(pp,2)==0 % parameter does not depend on ptype
        Parameters.(EstimParamNames{nEstimParamsFinder(pp,1)})=reshape(currparamraw,estimparamssizes(pp,:));
    else
        Parameters.(EstimParamNames{nEstimParamsFinder(pp,1)}).(Names_i{nEstimParamsFinder(pp,2)})=reshape(currparamraw,estimparamssizes(pp,:));
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters (including the masses of the permanent types)
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Solve the model, and evaluate the log-likelihood
% (the shock grids are set up inside these commands, so shocks that depend on the estimated parameters are always current)
[~, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

[LogL,LogLi,PTypePosterior]=LogLikelihood_FHorz_PType(Data,MeasurementError,jequaloneDist,PTypeDistParamNames,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid,pi_z,simoptions,estimoptions);

%% The objective
if estimoptions.vectoroutput==1
    Obj=LogLi;
elseif estimoptions.vectoroutput==2
    Obj=[LogLi, PTypePosterior];
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
