function AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)

if ~exist('simoptions','var')
    simoptions.lowmemory=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
else
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    end
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
end


%%
l_a=length(n_a);
N_a=prod(n_a);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions);


%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyIndexes,1);

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        FnsToEvalNames=simoptions.FnsToEvalNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%% I want to do some things now, so that they can be used in setting up conditional restrictions
AllStats=struct();

l_daprime=size(PolicyIndexes,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_z==0
    if simoptions.lowmemory==0
        StationaryDist=reshape(StationaryDist,[N_a,N_j]);
        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)
    else
        StationaryDist=reshape(StationaryDist,[N_a,N_j]);
        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    end
else
    if simoptions.lowmemory==0
        StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
        StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    else
        StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    end
end
% StationaryDistVec=gpuArray(StationaryDistVec);
% PolicyIndexes=gpuArray(PolicyIndexes);
%
% PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
% permuteindexes=[1+(1:1:(l_a+l_z)),1];
% PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]


%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it). 
if isfield(simoptions,'conditionalrestrictions')    
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionStruct=struct();

    % For each conditional restriction, create a 'restricted stationary distribution'
    for rr=1:length(CondlRestnFnNames)
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}));
        if length(temp)>(l_daprime+l_a+l_z)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end
        % Get parameter values for Conditional Restriction functions
        if isempty(CondlRestnFnParamNames) % check for '={}'
            CondlRestnFnParamsVec=[];
        else
            CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames);
        end
        % Store the actual functions
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});

        CondlRestnFnParamsCell=cell(length(CondlRestnFnParamsVec),1);
        for pp=1:length(CondlRestnFnParamsVec)
            CondlRestnFnParamsCell(pp,1)={CondlRestnFnParamsVec(pp)};
        end

        RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J));
        if N_z==0
            RestrictionValues=reshape(RestrictionValues,[N_a,N_j]);
        else
            RestrictionValues=reshape(RestrictionValues,[N_a*N_z,N_j]);
        end

        RestrictedStationaryDistVec=StationaryDist;
        RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction
        
        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr)=sum(RestrictedStationaryDistVec(:));
        RestrictionStruct(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass(rr);

        if restrictedsamplemass(rr)==0
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Just return this and hopefully it is clear to the user
        else
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Seems likely this would be something user might want
        end

    end
end


%%
if N_z==0
    if simoptions.lowmemory==0

        % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
        % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
        % PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)
        % 
        % a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after N_a
                end
            end
            
            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
            AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if restrictedsamplemass(rr)>0
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                    end
                end
            end
        end

    elseif simoptions.lowmemory==1 % Loop over age j

        % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
        % 
        % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);

        for ff=1:length(FnsToEvaluate)
            Values=nan(N_a,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,0,a_grid,[]);
            end
            AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if restrictedsamplemass(rr)>0
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec(:,jj),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                    end
                end
            end
        end
    end

else % N_z

    if simoptions.lowmemory==0

        StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
        % 
        % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
        % PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
        % 
        % a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)
            % Values=nan(N_a,N_z,N_j,'gpuArray');

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-2)}; % (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
                end
            end

            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
            AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if restrictedsamplemass(rr)>0
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                    end
                end
            end
        end

    elseif simoptions.lowmemory==1 % Loop over age j

        % StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
        % 
        % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);

        for ff=1:length(FnsToEvaluate)
            Values=nan(N_a*N_z,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_grid,z_gridvals_J(:,:,jj));
            end
            AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if restrictedsamplemass(rr)>0
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec(:,jj),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                    end
                end
            end
        end
    end
end


end
