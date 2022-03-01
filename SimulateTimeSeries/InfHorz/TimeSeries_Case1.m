function TimeSeries=TimeSeries_Case1(Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, d_grid, a_grid, z_grid,pi_z,simoptions)
% Simulate time series for the FnsToEvaluate for an infinite horizon model.

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

l_a=length(n_a);
l_z=length(n_z);

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    %If vfoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.burnin=1000;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10000;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'burnin')
        simoptions.burnin=1000;
    end
    if ~isfield(simoptions, 'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=10^4;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
end

%%
TimeSeriesIndexes=SimTimeSeriesIndexes_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    TimeSeriesNames=fieldnames(FnsToEvaluate);
    for ff=1:length(TimeSeriesNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(TimeSeriesNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(TimeSeriesNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
if simoptions.parallel~=2
    TimeSeriesIndexesKron=zeros(3,length(TimeSeriesIndexes(1,:)));
    
    for t=1:length(TimeSeriesIndexes(1,:))
        TimeSeriesIndexesKron(1,t)=sub2ind_homemade([n_a],TimeSeriesIndexes(1:l_a,t)); % a
        TimeSeriesIndexesKron(2,t)=sub2ind_homemade([n_z],TimeSeriesIndexes(l_a+1:l_a+l_z,t)); % z
    end
    
    TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, Policy,Parameters, FnsToEvaluate,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid);
    
    TimeSeries=TimeSeriesKron;
elseif simoptions.parallel==2
    TimeSeriesIndexes=gather(TimeSeriesIndexes);
    TimeSeriesIndexesKron=zeros(3,length(TimeSeriesIndexes(1,:)));
    
    for t=1:length(TimeSeriesIndexes(1,:))
        TimeSeriesIndexesKron(1,t)=sub2ind_homemade([n_a],TimeSeriesIndexes(1:l_a,t));%a
        TimeSeriesIndexesKron(2,t)=sub2ind_homemade([n_z],TimeSeriesIndexes(l_a+1:l_a+l_z,t));%z
    end
    
    TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, gather(Policy),Parameters, FnsToEvaluate,FnsToEvaluateParamNames, n_d, n_a, n_z, gather(d_grid), gather(a_grid), gather(z_grid));
    
    TimeSeries=gpuArray(TimeSeriesKron);

end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    TimeSeries2=TimeSeries;
    clear TimeSeries
    TimeSeries=struct();
%     TimeSeriesNames=fieldnames(FnsToEvaluate);
    for ff=1:length(TimeSeriesNames)
        TimeSeries.(TimeSeriesNames{ff})=TimeSeries2(ff,:);
    end
end

end