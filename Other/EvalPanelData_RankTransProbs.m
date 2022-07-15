function [TransitionProbabilities]=EvalPanelData_RankTransProbs(SimPanelValues,simoptions)
% Returns a Matrix 100-by-100 that contains the t-period transition probabilities for all of 
% the quantiles from 1 to 100 (for each variable in the panel data)
%
% NaN in the transition matrix indicates that the percentiles were not
% well defined (e.g., if you asked for npoints=5, it is trying to get the
% 20,40,60,80 and 100th percentiles. But if half the observations in
% SimPanelValues are one value and the other half are another value, then
% the difference between the 80 and 100 percentiles is ill-defined and you
% will get some NaN).
%
% Unless using simoptions.npoints in which case it will be npoints-by-npoints.
%%
if ~exist('simoptions','var')
    simoptions.npoints=100;
    simoptions.parallel=1;
else
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
    if ~isfield(simoptions,'paralle')
        simoptions.parallel=100;
    end    
end

%%
PanelVariableNames=fieldnames(SimPanelValues);

[T,NSims]=size(SimPanelValues.(PanelVariableNames{1})); % Number of time periods and number of simulations/individuals (in the panel data)

npoints=simoptions.npoints; % Reduce overhead on the parfor
% TransitionProbabilities.(PanelVariableNames{ff})=nan(npoints,npoints,T); % We first count the transitions, then normalize the rows
% Note the (:,:,T) entries will remain nan as the transitions from T to T+1 are not observed.

prctilestocalc=100*(1/npoints:1/npoints:1); % npoints is how many percentiles we want to calculate

for ff=1:length(PanelVariableNames) % Loop over the variables in the panel data
    Values=gather(SimPanelValues.(PanelVariableNames{ff})); % Make sure it is on cpu
    
    TransitionProbabilities_ff=zeros(npoints,npoints,T);
    
    % First, replace each of the values with its rank (conditional on time period)
	parfor tt=1:T
        Values_tt=Values(tt,:);
        P = prctile(Values_tt,prctilestocalc);
        if ff==2 && tt==15
            P
            [min(Values_tt),max(Values_tt)]
            [sum(Values_tt==min(Values_tt)),numel(Values_tt)]
        end
        for ii=1:NSims
            [~,Values_tt(ii)]=max(P>Values_tt(ii));
        end
        Values(tt,:)=Values_tt;
    end
    
    % Now, count all the rank transitions
    parfor tt=1:T-1  % Loop over the time periods except the last (as don't observe transitions out of the final period)
        Values_par=Values(tt:tt+1,:); % This is just to help matlab parallelize it (otherwise it couldn't splice both the tt and tt+1)
        Values_tt=Values_par(1,:)
        Values_ttplus1=Values_par(2,:);
        TransitionProbabilities_ff_tt=zeros(npoints,npoints);
        
        for ii=1:NSims
            TransitionProbabilities_ff_tt(Values_tt(ii),Values_ttplus1(ii))=TransitionProbabilities_ff_tt(Values_tt(ii),Values_ttplus1(ii))+1; % Add one to that transition
        end
        
        TransitionProbabilities_ff_tt=TransitionProbabilities_ff_tt./sum(TransitionProbabilities_ff_tt,2); % Normalise all the rows
        
        TransitionProbabilities_ff(:,:,tt)=TransitionProbabilities_ff_tt;
    end
    
    TransitionProbabilities.(PanelVariableNames{ff})=TransitionProbabilities_ff;
end

end