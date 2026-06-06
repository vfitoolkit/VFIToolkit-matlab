function [MatchedEV,DistMatches]=MEP_InfHorz_Step_MatchExpectations(VPath,N_a,N_z,N_S,T,pi_Sprime_T,AggVarsPath,SSprimemask_T,SSmask_T,SSprimemask_T_indexes,ss_ind_T,recursiveeqmoptions)
% VPath=reshape(VPath,[N_a,T,N_z]);

Distances=nan(T,T);
DistMatches=zeros(T,N_S,recursiveeqmoptions.matchE_nnearest,2); % Use the closest few expectations [4th dimension: index 1 contains the t-index for the matches, index 2 contains the distances of the matches]
% First, we have to find which agent dist with the same aggregate shock today are the 'most similar'
if recursiveeqmoptions.matchE_distmeasure==1 % Kolmogorov-Smirnoff distance
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
elseif recursiveeqmoptions.matchE_distmeasure==2 % Distance of first moments
    Distances_EndoState=100*abs(AggVarsPath.EndoState1.Mean'-AggVarsPath.EndoState1.Mean)./AggVarsPath.EndoState1.Mean; % percentage difference
    Distances_ExoState=100*abs(AggVarsPath.ExoState1.Mean'-AggVarsPath.ExoState1.Mean)./AggVarsPath.ExoState1.Mean; % percentage difference
    Distances=Distances_EndoState+Distances_ExoState;
end
Distances=gpuArray(Distances);
% Put indicators on the closest ones and store them in
MatchedEV_full=zeros(N_a,T,N_z,N_S,recursiveeqmoptions.matchE_nnearest,'gpuArray'); % note: N_a,T,N_z due to fastOLG
if recursiveeqmoptions.matchingsetup==1 % Match Sprime based on Distances
    for tt=1:T-1
        currSSprime=ss_ind_T(tt+1);
        % Need to look for periods that are the closest today in terms of having the same S and smallest Distances (same AgentDist), but having a different next period S
        for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
            if SSprime_c==currSSprime && tt>recursiveeqmoptions.burnin
                % The match is this period and distance is zero
                DistMatches(tt,SSprime_c,:,1)=tt;
                DistMatches(tt,SSprime_c,:,2)=0;
                % Put the next period value function into the MatchedEV
                MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,tt+1,:);
            else
                % SSmask_T(SS_c,:) is binary, and takes value of 1 for the time periods that have SS_c as the aggregate shock
                % notSSmask_Tplus1(SS_c,:) is binary, and takes value of 1 for the next-period time periods that do NOT have SS_c as the aggregate shock
                % potential=Distances(tt,:);
                % potential(notSSmask_Tplus1(SSprime_c,:))=Inf; % is making distance=Inf for anything that is not SSprime_c tomorrow
                potential=Distances(tt,SSprimemask_T(1,:,SSprime_c)); % Only consider distances that have (currSS,SSprime_c), that is, look for matching S, and then we find each of the Sprime (as we are looping over the Sprime)
                [matchval,matchind]=mink(potential,recursiveeqmoptions.matchE_nnearest); % Of the time periods that match S, and have the Sprime we are considering at the moment, find the closest as measured by 'Distances'
                % matchind is currently for the 1s in SSprimemask_T, convert to a tt index
                matchind=SSprimemask_T_indexes(matchind,SSprime_c);
                DistMatches(tt,SSprime_c,:,1)=matchind;
                DistMatches(tt,SSprime_c,:,2)=matchval;
                for nn=1:recursiveeqmoptions.matchE_nnearest
                    MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,matchind+1,:); % matchind+1 is where we are the period after the match [we want next period value fn to create expectations]
                end
            end
        end
    end
elseif recursiveeqmoptions.matchingsetup==2 % Match (S,Sprime) based on Distances
    for tt=1:T-1
        currSS=ss_ind_T(tt);
        currSSprime=ss_ind_T(tt+1);
        % Need to look for periods that are the closest today in terms of having the same S and smallest Distances (same AgentDist), but having a different next period S
        for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
            if SSprime_c==currSSprime && tt>recursiveeqmoptions.burnin
                % The match is this period and distance is zero
                DistMatches(tt,SSprime_c,:,1)=tt;
                DistMatches(tt,SSprime_c,:,2)=0;
                % Put the next period value function into the MatchedEV
                MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,tt+1,:);
            else
                % SSmask_T(SS_c,:) is binary, and takes value of 1 for the time periods that have SS_c as the aggregate shock
                % notSSmask_Tplus1(SS_c,:) is binary, and takes value of 1 for the next-period time periods that do NOT have SS_c as the aggregate shock
                % potential=Distances(tt,:);
                % potential(notSSmask_Tplus1(SSprime_c,:))=Inf; % is making distance=Inf for anything that is not SSprime_c tomorrow
                potential=Distances(tt,SSprimemask_T(1,:,currSS,SSprime_c)); % Only consider distances that have (currSS,SSprime_c), that is, look for matching S, and then we find each of the Sprime (as we are looping over the Sprime)
                [matchval,matchind]=mink(potential,recursiveeqmoptions.matchE_nnearest); % Of the time periods that match S, and have the Sprime we are considering at the moment, find the closest as measured by 'Distances'
                % matchind is currently for the 1s in SSprimemask_T, convert to a tt index
                matchind=SSprimemask_T_indexes(matchind,currSS,SSprime_c);
                DistMatches(tt,SSprime_c,:,1)=matchind;
                DistMatches(tt,SSprime_c,:,2)=matchval;
                for nn=1:recursiveeqmoptions.matchE_nnearest
                    MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,matchind+1,:); % matchind+1 is where we are the period after the match [we want next period value fn to create expectations]
                end
            end
        end
    end
end
% For period T, we just fill MatchedEV with the MatchedEV that is the period most similar in terms of S and 'Distances' [we cannot next period S, so this will do]
[~,matchindT]=min(Distances(T,:).*SSmask_T(ss_ind_T(T),:));
MatchedEV_full(:,T,:,:,:)=MatchedEV_full(:,matchindT,:,:,:);
% Now that we have the 'matches' we construct the expected next period value fn
MatchedEV=mean(MatchedEV_full,5); % First, take the mean over the 'nearest'
% MatchedEV is now [N_a,T,N_z,N_S]
% Second, take expectations with respect to next period S
MatchedEV=sum(MatchedEV.*pi_Sprime_T,4); % pi_Sprime_T is [1,T,1,N_S], note that S is determined by t, so only needs the Sprime probabilities and does not have a dimension for S
% MatchedEV is now [N_a,T,N_z]



end