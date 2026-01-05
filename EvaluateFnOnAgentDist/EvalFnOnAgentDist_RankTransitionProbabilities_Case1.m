function TransitionProbabilities=EvalFnOnAgentDist_RankTransitionProbabilities_Case1(t, NSims, StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, simoptions,npoints)
%Returns a Matrix 100-by-100 that contains the t-period transition probabilities for all of the quantiles from 1
%to 100. Unless the optional npoints input is used in which case it will be
%npoints-by-npoints. (third dimension is the different FnsToEvaluate that this
%matrix is calculated for)
%
% simoptions.parallel and npoints are optional inputs

%%
% In principle this could likely be done better based on cupola (rank
% transition matrix). For now just implementing a basic version based on
% simulation methods.

% To save overhead, move everything to cpu. Can then move back to gpu at
% end. Otherwise the subcodes would keep moving things back and forth as
% all simulations are done on cpu (in my experience this is always faster).

% We can just call the standard time series simulation codes to do this.
% Want no burnin, and to draw initial conditions randomly from
% StationaryDist. We only need to actually store the first and 't' period
% values. We can do this just based on index values for the state
% variables, and then just evaluate the functions on them and calculate the
% ranks at the end.

simoptions

%%
if ~isfield(simoptions,'parallel')
    simoptions.parallel=1+(gpuDeviceCount>0);
end
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;
end

if exist('npoints','var')==0
    npoints=100;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
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

%%
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

%% Start with generating starting indexes and using simulations to get finish/final indexes. Will give us NSims transitions (indexes).
simoptions.burnin=0;
simoptions.simperiods=t;
simoptions.numbersims=NSims;

% Switch to CPU for simulations
StationaryDist=gather(StationaryDist);

% Simulate a panel of t periods, and then just keep the first and last period from that
SimPanelValues=SimPanelValues_Case1(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions);

% cumsumStationaryDist=cumsum(reshape(StationaryDist,[N_a*N_z,1]));
% cumsum_pi_z=cumsum(pi_z,2);
% if simoptions.inheritanceasset==0
%     PolicyIndexesKron=gather(KronPolicyIndexes_Case1(Policy,n_d,n_a,n_z,simoptions));
%     Transitions_StartAndFinishIndexes=nan(2,NSims);
%     parfor ii=1:NSims
%         Transitions_StartAndFinishIndexes_ii=Transitions_StartAndFinishIndexes(:,ii);
%         % Draw initial condition
%         [~,seedpoint]=max(cumsumStationaryDist>rand(1,1));
%         Transitions_StartAndFinishIndexes_ii(1)=seedpoint;
%         seedpoint=ind2sub_homemade([N_a,N_z],seedpoint); % put in form needed for SimTimeSeriesIndexes_Case1_raw
%         % Simulate time series
%         SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyIndexesKron,l_d,n_a,cumsum_pi_z,seedpoint,simoptions);
%         % Store last
%         Transitions_StartAndFinishIndexes_ii(2)=sub2ind_homemade([N_a,N_z],SimTimeSeriesKron(:,end));
%         Transitions_StartAndFinishIndexes(:,ii)=Transitions_StartAndFinishIndexes_ii;
%     end
% end
% 
% %% Done simulating, switch back to GPU
% Transitions_StartAndFinishIndexes=gpuArray(Transitions_StartAndFinishIndexes);

%% We have the indexes. Now convert into values. Will give us NSims transitions (values).
Transitions_StartAndFinish=nan(2,NSims,length(FnsToEvaluate),'gpuArray');
Transitions_StartAndFinish_ff=nan(2,NSims,'gpuArray');

PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
permuteindexes=[1+(1:1:(l_a+l_z)),1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

for ff=1:length(FnsToEvalNames)
    % FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    % Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    % Values=reshape(Values,[N_a*N_z,1]);
    % % MIGHT BE POSSIBLE TO MERGE FOLLOWING TWO LINES (replace rows 1 and 2 with all rows ':'), JUST UNSURE WHAT
    % % RESULTING BEHAVIOUR WILL BE IN TERMS OF SIZE() AND CURRENTLY TO LAZY TO CHECK.
    % Transitions_StartAndFinish_ff(1,:)=Values(Transitions_StartAndFinishIndexes(1,:));
    % Transitions_StartAndFinish_ff(2,:)=Values(Transitions_StartAndFinishIndexes(2,:));
    temp=SimPanelValues.(FnsToEvalNames{ff});
    Transitions_StartAndFinish_ff(1,:)=temp(1,:);
    Transitions_StartAndFinish_ff(2,:)=temp(end,:);
    
    %% Now convert values into ranks. Will give us NSims transitions (ranks).
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_a*N_z,1]);
    % To do this we use the CDF, so start with calculating CDF for current 'SSvaluesFn'
    [SortedValues,SortedValues_index] = sort(Values);
    SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
    CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
    
    ranks_index_start=nan(NSims,1,'gpuArray');
    ranks_index_fin=nan(NSims,1,'gpuArray');
    parfor kk=1:NSims
        temp=Transitions_StartAndFinish_ff(:,kk);
        % Ranks of starting points
        [~,ranks_index_start(kk)]=max(SortedValues>temp(1));
        % Ranks of finishing points
        [~,ranks_index_fin(kk)]=max(SortedValues>temp(2));
    end
    
    % Ranks of starting points
    ranks=CumSumSortedStationaryDistVec(ranks_index_start);
    Transitions_StartAndFinish_ff(1,:)=ranks;
    % Ranks of finishing points
    ranks=CumSumSortedStationaryDistVec(ranks_index_fin);
    Transitions_StartAndFinish_ff(2,:)=ranks;
    
    Transitions_StartAndFinish(:,:,ff)=Transitions_StartAndFinish_ff;
end

%% We have NSims rank transitions. Now switch into rank transition probabilities.
TransitionProbabilities=nan(npoints,npoints,length(FnsToEvaluate),'gpuArray');
for ff=1:length(FnsToEvaluate)
    for i1=1:npoints
        denominator=nnz(Transitions_StartAndFinish(1,:,ff)==i1); % Starts with rank i1
        if denominator>0
            for i2=1:npoints
                numerator=sum(Transitions_StartAndFinish(1,:,ff)==i1 && Transitions_StartAndFinish(2,:,ff)==i2); % Starts with rank i1, ends with rank i2
                TransitionProbabilities(i1,i2,ff)=numerator/denominator;
            end
        end
    end
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    TransitionProbabilities2=TransitionProbabilities;
    clear TransitionProbabilities
    TransitionProbabilities=struct();
%     FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        TransitionProbabilities.(FnsToEvalNames{ff})=TransitionProbabilities2(:,:,ff);
    end
end


end
