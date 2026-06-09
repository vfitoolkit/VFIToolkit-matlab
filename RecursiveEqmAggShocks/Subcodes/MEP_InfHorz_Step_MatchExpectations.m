function [MatchedEV,DistMatches]=MEP_InfHorz_Step_MatchExpectations(VPath,N_a,N_z,l_a,l_z,N_S,T,pi_Sprime_T,AggVarsPath,matchexpectations,recursiveeqmoptions)
% VPath=reshape(VPath,[N_a,T,N_z]);

% MatchedEV will be [N_a,T,N_z] (as fastOLG)
% MatchedEV(:,t,:) is the 'matched expectation' for the next period E[V_{t+1}]
%
% We record the info about which periods are matched with which in
% DistMatches=zeros(T,N_S,recursiveeqmoptions.matchE_nnearest,2);
% DistMatches(:,:,:,1) contains the t-index for the matches
% DistMatches(:,:,:,2) contains the distances of the matches
% By default recursiveeqmoptions.matchE_nnearest=1, we use only one period as the match [we might, e.g. with =3, want to average the 3 closest matches]
% So,
% DistMatches(7,2,1,1) tells us the t-index for the period that is used as
% Sprime_c=2 for next-period EV in period 7; that is the expectation V_8(:,S_c=2) 
% that is used to compute E_{t=7}[V_{t=8}(:,S)].

%% First, we just measure which AgentDist are 'most similar'
% We measure the distance between them the AgentDist for every two periods
% Distances=zeros(T-1,T,'gpuArray');
% Distances(tt1,tt2)
% Interpretation: tt1 is this period
% We want to find the 'match' for the expecations period
% So tt2 will be 'next period'
% We think of this as, e.g., Distances(tt1,tt2) is the 'distance between K in period tt1+1 and period tt2'
% In this way, the minimum of Distances(tt1,:) is telling us the closest match for the expectations term that is needed to be form for period tt1 (which is about E_t[V_{t+1}]) 

if recursiveeqmoptions.matchE_distmeasure==1
    %% Kolmogorov-Smirnoff distance
    Distances=zeros(T-1,T,'gpuArray');
    % Need to parallelize
    % Calculate distance for the upper triangular
    for tt1=1:T
        for tt2=tt1+1:T
            % Compare AgentDistPath(:,:,tt1) with AgentDistPath(:,:,tt2)
            % dist=  % Note: fastOLG, so AgentDistPath is (a,j,z)-by-1
            error('Not yet implemented')
        end
        Distances(tt1,tt2)=dist;
    end
    % Fill in the lower triangular (distance is symmetric), put Inf on diagonal
    % PARALLELIZE THIS
    for tt1=1:T
        for tt2=1:tt1-1
            Distances(tt1,tt2)=Distances(tt2,tt1);
        end
        Distances(tt1,tt1)=Inf;
    end
    % NEED recursiveeqmoptions.matching_IdiosyncraticExogenousStates
    % NEED Distance_T

elseif recursiveeqmoptions.matchE_distmeasure==2 
    %% Percentage distance of first moments
    if l_a>=1
        % nextK=AggVarsPath.EndoState1.Mean(2:T)'; % K for next period, which is what we want to match to
        Distances_EndoState1=100*abs(AggVarsPath.EndoState1.Mean(2:T)'-AggVarsPath.EndoState1.Mean)./AggVarsPath.EndoState1.Mean(2:T)'; % percentage difference
        if l_a>=2
            % nextK=AggVarsPath.EndoState3.Mean(2:T)';
            Distances_EndoState2=100*abs(AggVarsPath.EndoState2.Mean(2:T)'-AggVarsPath.EndoState2.Mean)./AggVarsPath.EndoState2.Mean(2:T)'; % percentage difference
            if l_a>=3
                % nextK=AggVarsPath.EndoState3.Mean(2:T)';
                Distances_EndoState3=100*abs(AggVarsPath.EndoState3.Mean(2:T)'-AggVarsPath.EndoState3.Mean)./AggVarsPath.EndoState3.Mean(2:T)'; % percentage difference
                Distances_EndoState=(Distances_EndoState1+Distances_EndoState2+Distances_EndoState3)/3;
            else
                Distances_EndoState=(Distances_EndoState1+Distances_EndoState2)/2;
            end
        else
            Distances_EndoState=Distances_EndoState1;
        end
    end
    if recursiveeqmoptions.matching_IdiosyncraticExogenousStates==2
        % Exogenous states are determined in general eqm, so have to recompute matching distances every iteration of the path
        % SHOULD I MATCH TO THIS PERIOD OR NEXT PERIOD EXOGENOUS STATE???
        if l_z>=1
            Distances_ExoState1=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
            if l_z>=2
                Distances_ExoState2=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
                if l_z>=3
                    Distances_ExoState3=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
                    if l_z>=4
                        Distances_ExoState4=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
                        if l_z>=5
                            Distances_ExoState5=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
                            Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3+Distances_ExoState4+Distances_ExoState5)/5;
                        else
                            Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3+Distances_ExoState4)/4;
                        end
                    else
                        Distances_ExoState=(Distances_ExoState1+Distances_ExoState2+Distances_ExoState3)/3;
                    end
                else
                    Distances_ExoState=(Distances_ExoState1+Distances_ExoState2)/2;
                end
            else
                Distances_ExoState=Distances_ExoState1;
            end
        end
        Distances=Distances_EndoState+Distances_ExoState;
    elseif recursiveeqmoptions.matching_IdiosyncraticExogenousStates==1
        % Use the Distances_ExoState that was pre-computed as it does not change across the iterations of the path
        Distances=Distances_EndoState+matchexpectations.Distances_ExoState;
    elseif recursiveeqmoptions.matching_IdiosyncraticExogenousStates==0
        % Not using ExoState as part of the matching (most likely just because exogenous states are constant across periods anyway)
        Distances=Distances_EndoState;
    end
end
% We think of this as, e.g., Distances(tt1,tt2) is the 'distance between K in period tt1 and period tt2)

if recursiveeqmoptions.matchE_nnearest>1
    error('Not yet implemented recursiveeqmoptions.matchE_nnearest>1')
end


%% Put indicators on the closest ones and store them in
DistMatches=zeros(T-1,N_S,recursiveeqmoptions.matchE_nnearest,2,'gpuArray');
MatchedEV_full=zeros(N_a,T,N_z,N_S,recursiveeqmoptions.matchE_nnearest,'gpuArray'); % note: N_a,T,N_z due to fastOLG

% if recursiveeqmoptions.matchingsetup==1 % Match Sprime based on Distances in t+1
% if recursiveeqmoptions.matchingsetup==2 % Match (S,Sprime) based on Distances
% NOTE: THIS IS ACTUALLY THE SAME CODE. WE CAN DEAL WITH THE 'Match (S,Sprime) via DistancesModifier

for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
    Sstr=['S',num2str(SSprime_c),'distancemodifier'];
    % For each period t, we want to find the t+1 match that we can use to create E[V_{t+1}]

    % Put 'zero' where SSprime_c is already the next period
    % Put 'Inf' where SSprime is not equal to SSprime_c, so they are out of contention
    % Following step does both these
    Distances_S=matchexpectations.(Sstr).*Distances;

    % Get the closest of the contenders
    [DistMatches_val,DistMatches_ind]=min(Distances_S,[],2);
    DistMatches(:,SSprime_c,1,1)=DistMatches_ind;
    DistMatches(:,SSprime_c,1,2)=DistMatches_val;

    % Put the next period value function into the MatchedEV
    MatchedEV_full(:,1:T-1,:,SSprime_c,1)=VPath(:,DistMatches_ind,:);
end


%% Build MatchedEV itself
% Now that we have the 'matches' we construct the expected next period value fn
MatchedEV=mean(MatchedEV_full,5); % First, take the mean over the 'nearest'

% MatchedEV is now [N_a,T,N_z,N_S]
% Second, take expectations with respect to next period S
MatchedEV=sum(MatchedEV.*pi_Sprime_T,4); % pi_Sprime_T is [1,T,1,N_S], note that S is determined by t, so only needs the Sprime probabilities and does not have a dimension for S
% MatchedEV is now [N_a,T,N_z]


% Lastly we fill in for the period T 
% Just use the expectations term we have built for the time period that looks most like time period T looks
[~,matchindT]=min(Distances(T-1,:)); % Because of how distances is construced, Distances(T-1,:) is the endo state for the expectation, which happens to be the period T itself
MatchedEV(:,T,:)=MatchedEV(:,matchindT,:);
% [MAYBE I SHOULD FILL IN THE FIRST AND LAST burnin PERIODS HERE AS WELL? 
% Although currently all but the actual S_{t+1} are already blocked from
% coming from the burnin periods. So probably little difference.]





end