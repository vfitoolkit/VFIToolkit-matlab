function [TransitionProbabilities]=SSvalues_RankTransitionProbabilities_Case1(t, NSims, StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, parallel,npoints)
%Returns a Matrix 100-by-100 that contains the t-period transition probabilities for all of the quantiles from 1
%to 100. Unless the optional npoints input is used in which case it will be
%npoints-by-npoints. (third dimension is the different SSvalueFn that this
%matrix is calculated for)

if nargin<14
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

%%
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

MoveOutputToGPU=0;
if parallel==2  
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexes=gather(PolicyIndexes);
    StationaryDist=gather(StationaryDist);
    MoveOutputToGPU=1;
end
parallel=1; % Want to do the simulations on parallel CPU, so switch to this


%% Start with generating starting indexes and using simulations to get finish/final indexes. Will give us NSims transitions (indexes).
burnin=0;
simperiods=t;

% NSims=10^5;

cumsumStationaryDist=cumsum(reshape(StationaryDist,[N_a*N_z,1]));
% [~,seedpoints]=max(cumsumStationaryDist>rand(1,NSims,'gpuArray'));

Transitions_StartAndFinishIndexes=nan(2,NSims);
parfor ii=1:NSims
    Transitions_StartAndFinishIndexes_ii=Transitions_StartAndFinishIndexes(:,ii);
    % Draw initial condition
    [~,seedpoint]=max(cumsumStationaryDist>rand(1,1,'gpuArray'));
    Transitions_StartAndFinishIndexes_ii(1)=seedpoint;
    seedpoint=ind2sub_homemade([N_a,N_z],seedpoint); % put in form needed for SimTimeSeriesIndexes_Case1_raw
    % Simulate time series
    SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z,burnin,seedpoint,simperiods,parallel);
    % Store last
    Transitions_StartAndFinishIndexes_ii(2)=sub2ind_homemade([N_a,N_z],SimTimeSeriesKron(:,end));
    Transitions_StartAndFinishIndexes(:,ii)=Transitions_StartAndFinishIndexes_ii;
end

%% Done simulating, switch back to GPU
if MoveOutputToGPU==1
    Transitions_StartAndFinishIndexes=gpuArray(Transitions_StartAndFinishIndexes);
    PolicyIndexes=gpuArray(PolicyIndexes);
end

%% We have the indexes. Now convert into values. Will give us NSims transitions (values).
Transitions_StartAndFinish=nan(2,NSims,length(SSvaluesFn),'gpuArray');
Transitions_StartAndFinish_jj=nan(2,NSims,'gpuArray');

PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, parallel);
permuteindexes=[1+(1:1:(l_a+l_z)),1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

for jj=1:length(SSvaluesFn)
    % Includes check for cases in which no parameters are actually required
    if isempty(SSvalueParamNames(jj).Names)  % check for 'SSvalueParamNames={}'
        SSvalueParamsVec=[];
    else
        SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(jj).Names);
    end
    Values=ValuesOnSSGrid_Case1(SSvaluesFn{jj}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,parallel);
    Values=reshape(Values,[N_a*N_z,1]);
    % MIGHT BE POSSIBLE TO MERGE FOLLOWING TWO LINES (replace rows 1 and 2 with all rows ':'), JUST UNSURE WHAT
    % RESULTING BEHAVIOUR WILL BE IN TERMS OF SIZE() AND CURRENTLY TO LAZY TO CHECK.
    Transitions_StartAndFinish_jj(1,:)=Values(Transitions_StartAndFinishIndexes(1,:));
    Transitions_StartAndFinish_jj(2,:)=Values(Transitions_StartAndFinishIndexes(2,:));
    
    %% Now convert values into ranks. Will give us NSims transitions (ranks).
    % Do this inside same for loop as also uses Values.
    % To do this we use the CDF, so start with calculating CDF for current 'SSvaluesFn'
    [SortedValues,SortedValues_index] = sort(Values);
    SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
    CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
    
    ranks_index_start=nan(NSims,1,'gpuArray');
    ranks_index_fin=nan(NSims,1,'gpuArray');
    parfor kk=1:NSims
        temp=Transitions_StartAndFinish_jj(:,kk);
        % Ranks of starting points
        [~,ranks_index_start(kk)]=max(SortedValues>temp(1));
        % Ranks of finishing points
        [~,ranks_index_fin(kk)]=max(SortedValues>temp(2));
    end
    
    % Ranks of starting points
    ranks=CumSumSortedStationaryDistVec(ranks_index_start);
    Transitions_StartAndFinish_jj(1,:)=ranks;
    % Ranks of finishing points
    ranks=CumSumSortedStationaryDistVec(ranks_index_fin);
    Transitions_StartAndFinish_jj(2,:)=ranks;
    
    Transitions_StartAndFinish(:,:,jj)=Transitions_StartAndFinish_jj;
end

%% We have NSims rank transitions. Now switch into rank transition probabilities.
TransitionProbabilities=nan(npoints,npoints,length(SSvaluesFn),'gpuArray');
for jj=1:length(SSvaluesFn)
    for i1=1:npoints
        denominator=nnz(Transitions_StartAndFinish(1,:,jj)==i1); % Starts with rank i1
        if denominator>0
            for i2=1:npoints
                numerator=sum(Transitions_StartAndFinish(1,:,jj)==i1 && Transitions_StartAndFinish(2,:,jj)==i2); % Starts with rank i1, ends with rank i2
                TransitionProbabilities(i1,i2,jj)=numerator/denominator;
            end
        end
    end
end



end
