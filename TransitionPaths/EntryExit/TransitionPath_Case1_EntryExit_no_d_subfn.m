function WeightedSumSq_GeneralEqmCondnPath=TransitionPath_Case1_EntryExit_no_d_subfn(PricePathVec, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, EntryExitParamNames, transpathoptions, vfoptions, simoptions)

unkronoptions.parallel=2;

N_z=prod(n_z);
N_a=prod(n_a);

l_p=length(PricePathNames);
PricePathOld=reshape(PricePathVec,[T,l_p]);

PricePathDist=Inf;
pathcounter=1;

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial.pdf=reshape(AgentDist_initial.pdf,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
% GeneralEqmCondnPath=zeros(size(PricePathOld),'gpuArray'); GeneralEqmCondnPath(T,:)=PricePathOld(T,:);
GeneralEqmCondnPath=ones(T,length(GeneralEqmEqns),'gpuArray'); GeneralEqmCondnPath(T,:)=zeros(1,length(GeneralEqmEqns),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');
PolicyWhenExit=zeros(N_a,N_z,'gpuArray');  % Will only be used when vfoptions.endogenousexit==2


if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%% Figure out what kind of 'general eqm conditions' are being used.
% Figure out which general eqm conditions are normal
GeneralEqmConditionsVec=zeros(1,length(GeneralEqmEqns));
standardgeneqmcondnsused=0;
specialgeneqmcondnsused=0;
entrycondnexists=0; condlentrycondnexists=0;
if ~isfield(transpathoptions,'specialgeneqmcondn')
    standardgeneqmcondnindex=1:1:length(GeneralEqmEqns);
else
    standardgeneqmcondnindex=zeros(1,length(GeneralEqmEqns));
    jj=1;
    GeneralEqmEqnParamNames_Full=GeneralEqmEqnParamNames;
    clear GeneralEqmEqnParamNames
    for ii=1:length(GeneralEqmEqns)
        if isnumeric(transpathoptions.specialgeneqmcondn{ii}) % numeric means equal to zero and is a standard GEqm
            standardgeneqmcondnsused=1;
            standardgeneqmcondnindex(jj)=ii;
            GeneralEqmEqnParamNames(jj).Names=GeneralEqmEqnParamNames_Full(ii).Names;
            jj=jj+1;
        elseif strcmp(transpathoptions.specialgeneqmcondn{ii},'entry')
            specialgeneqmcondnsused=1;
            entrycondnexists=1;
            % currently 'entry' is the only kind of specialgeneqmcondn
            entrygeneqmcondnindex=ii;
            EntryCondnEqn=GeneralEqmEqns(ii);
            EntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        elseif strcmp(transpathoptions.specialgeneqmcondn{ii},'condlentry')
            specialgeneqmcondnsused=1;
            condlentrycondnexists=1;
            condlentrygeneqmcondnindex=ii;
            CondlEntryCondnEqn=GeneralEqmEqns(ii);
            CondlEntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
            if condlentrygeneqmcondnindex~=length(GeneralEqmEqns)
                fprintf('ERROR: when using condlentry in transition paths it must be the last GeneralEqmEqn \n')
                break
                return
            end
        end
    end
    standardgeneqmcondnindex=standardgeneqmcondnindex(standardgeneqmcondnindex>0); % get rid of zeros at the end
    GeneralEqmEqns=GeneralEqmEqns(standardgeneqmcondnindex);
end

%% 
PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
if vfoptions.endogenousexit==1
    ExitPolicyPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
end
if vfoptions.endogenousexit==2
    PolicyWhenExitPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
end
if entrycondnexists==1
    VPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for i=1:T-1 %so t=T-i
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-i,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-i,kk);
    end
    
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
    beta=prod(DiscountFactorParamsVec);
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid,ReturnFnParamsVec);
    
    if vfoptions.endogenousexit>0
        % The 'return to exit function' parameters (in order)
        ReturnToExitFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ReturnToExitFnParamNames);
        %             if vfoptions.returnmatrix==0
        %                 ReturnToExitMatrix=CreateReturnToExitFnMatrix_Case1_Disc(vfoptions.ReturnToExitFn, n_a, n_z, a_grid, z_grid, vfoptions.parallel, ReturnToExitFnParamsVec);
        %             elseif vfoptions.returnmatrix==1
        %                 ReturnToExitMatrix=vfoptions.ReturnToExitFn; % It is simply assumed that you are doing this for both.
        %             elseif vfoptions.returnmatrix==2 % GPU
        if vfoptions.endogenousexit==1
            ReturnToExitMatrix=CreateReturnToExitFnMatrix_Case1_Disc_Par2(vfoptions.ReturnToExitFn, n_a, n_z, a_grid, z_grid, ReturnToExitFnParamsVec);
        elseif vfoptions.endogenousexit==2
            ReturnToExitMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(vfoptions.ReturnToExitFn, 0, n_a, n_z, [], a_grid, z_grid, ReturnToExitFnParamsVec);
            continuationcost=CreateVectorFromParams(Parameters, vfoptions.endogenousexitcontinuationcost);
            exitprobabilities=CreateVectorFromParams(Parameters, vfoptions.exitprobabilities);
            exitprobs=[1-sum(exitprobabilities),exitprobabilities];
        end
        %             end
    end
    
    if vfoptions.endogenousexit==0
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,z_c)=Vtemp;
            Policy(:,z_c)=maxindex;
            
        end
    elseif vfoptions.endogenousexit==1
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            ReturnToExitMatrix_z=ReturnToExitMatrix(:,z_c);
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by a
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            % Exit decision
            ExitPolicy(:,z_c)=((ReturnToExitMatrix_z-Vtemp')>0); % Assumes that when indifferent you do not exit.
            V(:,z_c)=ExitPolicy(:,z_c).*ReturnToExitMatrix_z+(1-ExitPolicy(:,z_c)).*Vtemp';
            Policy(:,z_c)=maxindex; % Note that this includes the policy that would be chosen if you did
            % not exit, even when choose exit. This is because it makes it much easier to then implement
            % Howards, and can just impose the =0 on exit on the final PolicyIndexes at the end of this
            % function just prior to output.
            
            %                 tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
            %                 Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
        end
    elseif vfoptions.endogenousexit==2
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            ReturnToExitMatrix_z=ReturnToExitMatrix(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but not on control variables
            EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by a
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            % Calc the max and it's index when exiting
            [FtempWhenExit,maxindexWhenExit]=max(ReturnToExitMatrix_z,[],1); % MOVE THIS OUTSIDE OF THE while loop
            % Endogenous Exit decision
            ExitPolicy_z=((FtempWhenExit-(Vtemp-continuationcost))>0); % Assumes that when indifferent you do not exit.
            
            % % The following line is implementing in a single line what is commented out here.
            % V_z_noexit=Vtemp;
            % V_z_endogexit=ExitPolicy(:,z_c).*FtempWhenExit+(1-ExitPolicy(:,z_c)).*(Vtemp-continuationcost);
            % V_z_exoexit=ReturnToExitMatrix_z;
            % VKron(:,z_c)=exitprobabilities(1)*V_z_noexit+exitprobabilities(2)*V_z_endoexit+exitprobabilities(3)*V_z_exoexit
            
            V(:,z_c)=exitprobs(1)*Vtemp+exitprobs(2)*(ExitPolicy_z.*FtempWhenExit+(1-ExitPolicy_z).*(Vtemp-continuationcost))+exitprobs(3)*FtempWhenExit;
            %                 FWhenExit(:,z_c)=FtempWhenExit;
            Policy(:,z_c)=maxindex;
            PolicyWhenExit(:,z_c)=maxindexWhenExit;
            ExitPolicy(:,z_c)=ExitPolicy_z;
            
            %                 tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
            %                 Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
        end
    end
    
    PolicyIndexesPath(:,:,T-i)=Policy;
    if vfoptions.endogenousexit>0
        ExitPolicyPath(:,:,T-i)=ExitPolicy; %Periods 1 to T-1
    end
    if vfoptions.endogenousexit==2
        PolicyWhenExitPath(:,:,T-i)=PolicyWhenExitPolicy; %Periods 1 to T-1
    end
    if entrycondnexists==1
        VPath(:,:,T-i)=V;
    end
    Vnext=V;
end
% Free up space on GPU by deleting things no longer needed
clear ReturnMatrix ReturnMatrix_z entireRHS EV_z Vtemp maxindex V Vnext

%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
%Call SteadyStateDist the current periods distn and SteadyStateDistnext
%the next periods distn which we must calculate
AgentDist=AgentDist_initial.mass*AgentDist_initial.pdf;
for i=1:T-1
    %Get the current optimal policy
    Policy=PolicyIndexesPath(:,:,i);
    if vfoptions.endogenousexit>0
        ExitPolicy=ExitPolicyPath(:,:,i); %Periods 1 to T-1
    end
    if vfoptions.endogenousexit==2
        PolicyWhenExit=PolicyWhenExitPath(:,:,i); %Periods 1 to T-1
    end
    if entrycondnexists==1
        V=VPath(:,:,i);
    end
    
    % Get the Entry-Exit parameters out of Parameters.
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(i,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(i,kk);
    end
    if vfoptions.endogenousexit==1 || vfoptions.endogenousexit==2
        CondlProbOfSurvival=1-ExitPolicy;
    else
        CondlProbOfSurvival=Parameters.(EntryExitParamNames.CondlProbOfSurvival{1});
    end
    %         DistOfNewAgents=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
    %         MassOfNewAgents=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
    %         % StationaryDistKron.mass
    %         % StationaryDistKron.pdf
    
    % Entry parameters out of Parameters
    DistOfNewAgents=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
    MassOfNewAgents=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
    % Conditional Entry means these are actually the potential entrants, so need to make the following changes.
    if condlentrycondnexists==1
        % Evaluate the conditional equilibrium condition on the (potential entrants) grid,
        % and where it is >=0 use this to set new values for the
        % EntryExitParamNames.CondlEntryDecisions parameter.
        CondlEntryCondnEqnParamsVec=CreateVectorFromParams(Parameters, CondlEntryCondnEqnParamNames(1).Names);
        CondlEntryCondnEqnParamsCell=cell(length(CondlEntryCondnEqnParamsVec),1);
        for jj=1:length(CondlEntryCondnEqnParamsVec)
            CondlEntryCondnEqnParamsCell(jj,1)={CondlEntryCondnEqnParamsVec(jj)};
        end
        Parameters.(EntryExitParamNames.CondlEntryDecisions{1})=(CondlEntryCondnEqn{1}(V,GEprices,CondlEntryCondnEqnParamsCell{:}) >=0);
        DistOfNewAgents=DistOfNewAgents.*Parameters.(EntryExitParamNames.CondlEntryDecisions{1});
    end
    
    % Check whether CondlProbOfSurvival is a matrix, or scalar, and act accordingly.
    if isscalar(gather(CondlProbOfSurvival))
        % No need to do anything
    elseif isa(gather(CondlProbOfSurvival),'numeric')
        CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
    else % Does not appear to have been inputted correctly
        fprintf('ERROR: CondlProbOfSurvival parameter does not appear to have been inputted with correct format \n')
        dbstack
        return
    end
    % Move these to where they need to be.
    if simoptions.parallel==2 % On GPU
        DistOfNewAgentsKron=reshape(gpuArray(DistOfNewAgents),[N_a*N_z,1]);
        CondlProbOfSurvival=gpuArray(CondlProbOfSurvival);
    elseif simoptions.parallel<2 % On CPU
        DistOfNewAgentsKron=reshape(gather(DistOfNewAgents),[N_a*N_z,1]);
        CondlProbOfSurvival=gather(CondlProbOfSurvival);
    elseif simoptions.parallel==3 % On CPU, sparse matrix
        DistOfNewAgentsKron=reshape(sparse(gather(DistOfNewAgents)),[N_a*N_z,1]);
        CondlProbOfSurvival=sparse(gather(CondlProbOfSurvival));
    end
    % Note: CondlProbOfSurvival is [N_a*N_z,1] because it will multiply Ptranspose.
    
    optaprime=reshape(Policy,[1,N_a*N_z]);
    %         if simoptions.endogenousexit==1
    %             optaprime=optaprime+(1-CondlProbOfSurvival'); % endogenous exit means that CondlProbOfSurvival will be 1-ExitPolicy
    %             % This will make all those who 'exit' instead move to first point on
    %             % 'grid on a'. Since as part of it's creation Ptranspose then gets multiplied by the
    %             % CondlProbOfSurvival these agents will all 'die' anyway.
    %             % It is done as otherwise the optaprime policy is being stored as
    %             % 'zero' for those who exit, and this causes an error when trying to
    %             % use optaprime as an index.
    %             % (Need to use transpose of CondlProbOfSurvival because it is being
    %             % kept in the 'transposed' form as usually is used to multiply Ptranspose.)
    %         end
    
    %%
    if simoptions.endogenousexit==0
        if simoptions.parallel<2
            Ptranspose=zeros(N_a,N_a*N_z);
            Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        elseif simoptions.parallel==2 % Using the GPU
            Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
            Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        elseif simoptions.parallel>2
            Ptranspose=sparse(N_a,N_a*N_z);
            Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        end
    elseif simoptions.endogenousexit==1
        if simoptions.parallel<2
            Ptranspose=zeros(N_a,N_a*N_z);
            temp=optaprime+N_a*(0:1:N_a*N_z-1);
            temp=temp(optaprime>0); % temp is just optaprime conditional on staying
            Ptranspose(temp)=1;
            %         Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        elseif simoptions.parallel==2 % Using the GPU
            Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
            temp=optaprime+N_a*(gpuArray(0:1:N_a*N_z-1));
            temp=temp(optaprime>0); % temp is just optaprime conditional on staying
            Ptranspose(temp)=1;
            %         if simoptions.endogenousexit==1 % I originally used unique, but
            %         then realised that the above three lines are a much more computationally
            %         efficient way to acheive the same thing.
            %             Ptranspose(unique(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1))))=1; % Relates to part of code above when creating optaprime. See explanation there for why have to treat these two cases seperately and add 'unique()' for this case.
            %         else
            %             Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            %         end
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        elseif simoptions.parallel>2
            Ptranspose=sparse(N_a,N_a*N_z);
            temp=optaprime+N_a*(0:1:N_a*N_z-1);
            temp=temp(optaprime>0); % temp is just optaprime conditional on staying
            Ptranspose(temp)=1;
            %         Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
            else
                Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            end
        end
    elseif simoptions.endogenousexit==2
        exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
        exitprobs=[1-sum(exitprobabilities),exitprobabilities];
        % Mixed exit (endogenous and exogenous), so we know that CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
        if simoptions.parallel<2
            Ptranspose=zeros(N_a,N_a*N_z);
            Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            %         Ptranspose1=(kron(pi_z',ones(N_a,N_a))).*(kron(exitprob(1)*ones(N_z,1),Ptranspose)); % No exit, and remove exog exit
            %         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            %         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
            % Following line does (in one line) what the above three commented
            % out lines do (doing it in one presumably reduces memory usage of Ptranspose1 and Ptranspose2)
            Ptranspose=((kron(pi_z',ones(N_a,N_a))).*(kron(exitprobs(1)*ones(N_z,1),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
        elseif simoptions.parallel==2 % Using the GPU
            exitprobs=gpuArray(exitprobs);
            Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
            Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            %         Ptranspose1=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(exitprob(1)*ones(N_z,1,'gpuArray'),Ptranspose)); % No exit, and remove exog exit
            %         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            %         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
            Ptranspose=((kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(exitprobs(1)*ones(N_z,1,'gpuArray'),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
        elseif simoptions.parallel>2
            Ptranspose=sparse(N_a,N_a*N_z);
            Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
            %         Ptranspose1=(kron(pi_z',ones(N_a,N_a))).*(kron(exitprob(1)*ones(N_z,1),Ptranspose)); % No exit, and remove exog exit
            %         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
            %         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
            Ptranspose=((kron(pi_z',ones(N_a,N_a))).*(kron(exitprobs(1)*ones(N_z,1),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
        end
    end
    %%
    %         % Create Ptranspose
    %         if simoptions.parallel<2
    %             Ptranspose=zeros(N_a,N_a*N_z);
    %             Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    %             if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
    %             else
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
    %             end
    %         elseif simoptions.parallel==2 % Using the GPU
    %             Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
    %             Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    %             if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose));
    %             else
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
    %             end
    %         elseif simoptions.parallel>2
    %             Ptranspose=sparse(N_a,N_a*N_z);
    %             Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    %             if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
    %             else
    %                 Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
    %             end
    %         end
    %         Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
    %         Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    %         Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
    
    AgentDistnext=MassOfNewAgents*DistOfNewAgentsKron+Ptranspose*AgentDist;
    %         AgentDistnext=Ptran*AgentDist;
    
    p=PricePathOld(i,:);
    
    for nn=1:length(ParamPathNames)
        Parameters.(ParamPathNames{nn})=ParamPath(i,nn);
    end
    for nn=1:length(PricePathNames)
        Parameters.(PricePathNames{nn})=PricePathOld(i,nn);
    end
    
    PolicyTemp=UnKronPolicyIndexes_Case1(Policy, 0, n_a, n_z,unkronoptions);
    if vfoptions.endogenousexit==2
        PolicyWhenExitTemp=UnKronPolicyIndexes_Case1(PolicyWhenExitTemp, 0, n_a, n_z,unkronoptions);
    end
    
    % Turn it into the 'mass and pdf' format used by EvaluateFnOnDist commands
    AgentDisttemp.mass=sum(sum(AgentDistnext));
    AgentDisttemp.pdf=AgentDistnext/AgentDisttemp.mass;
    if simoptions.parallel>=3 % Solve with sparse matrix
        AgentDisttemp.pdf=full(AgentDisttemp.pdf);
        if simoptions.parallel==4 % Solve with sparse matrix, but return answer on gpu.
            AgentDisttemp.pdf=gpuArray(AgentDisttemp.pdf);
        end
    end
    
    AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDisttemp, PolicyTemp, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, 0, n_a, n_z, [], a_grid, z_grid, simoptions.parallel,simoptions,EntryExitParamNames, PolicyWhenExitTemp);
    
    
    % Evaluate all the general eqm conditions for the current period
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    % GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));
    if standardgeneqmcondnsused==1
        % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
        GeneralEqmConditionsVec(standardgeneqmcondnindex)=gather(real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel)));
    end
    % Now fill in the 'non-standard' cases
    if specialgeneqmcondnsused==1
        if condlentrycondnexists==1
            %                 % Evaluate the conditional equilibrium condition on the (potential entrants) grid,
            %                 % and where it is >=0 use this to set new values for the
            %                 % EntryExitParamNames.CondlEntryDecisions parameter.
            %                 CondlEntryCondnEqnParamsVec=CreateVectorFromParams(Parameters, CondlEntryCondnEqnParamNames(1).Names);
            %                 CondlEntryCondnEqnParamsCell=cell(length(CondlEntryCondnEqnParamsVec),1);
            %                 for jj=1:length(CondlEntryCondnEqnParamsVec)
            %                     CondlEntryCondnEqnParamsCell(jj,1)={CondlEntryCondnEqnParamsVec(jj)};
            %                 end
            %
            %                 Parameters.(EntryExitParamNames.CondlEntryDecisions{1})=(CondlEntryCondnEqn{1}(V,p,CondlEntryCondnEqnParamsCell{:}) >=0);
            %                 GeneralEqmConditionsVec(condlentrygeneqmcondnindex)=0; % Because the EntryExitParamNames.CondlEntryDecisions is set to hold exactly we can consider this as contributing 0
            if entrycondnexists==1
                % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
                EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.CondlEntryDecisions{1}),[numel(V),1]));
                % @(EValueFn,ce)
                % And use entrants distribution, not the stationary distn
                GeneralEqmConditionsVec(entrygeneqmcondnindex)=gather(real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel)));
            end
        else
            if entrycondnexists==1
                % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
                EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]));
                % @(EValueFn,ce)
                % And use entrants distribution, not the stationary distn
                GeneralEqmConditionsVec(entrygeneqmcondnindex)=gather(real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel)));
            end
        end
        %     if entrycondnexists==1
        %         % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
        %         EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]));
        %         % @(EValueFn,ce)
        %         % And use entrants distribution, not the stationary distn
        %         GeneralEqmConditionsVec(entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
        %     end
    end
    
%     % When using negative powers matlab will often return complex
%     % numbers, even if the solution is actually a real number. I
%     % force converting these to real, albeit at the risk of missing problems
%     % created by actual complex numbers.
%     if transpathoptions.GEnewprice==1
%         %             PricePathNew(i,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
        if condlentrycondnexists==0
            GeneralEqmCondnPath(i,:)=GeneralEqmConditionsVec;
        elseif condlentrycondnexists==1
            GeneralEqmCondnPath(i,:)=GeneralEqmConditionsVec(1:end-1); % The conditional entry condition is required to be last when doing transition paths
        end
%     elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
%         fprintf('ERROR: transpathoptions.GEnewprice==0 NOT YET IMPLEMENTED (TransitionPath_Case1_no_d.m)')
%         return
%         for j=1:length(GeneralEqmEqns)
%             GEeqn_temp=@(p) real(MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParamsVec));
%             PricePathNew(i,j)=fzero(GEeqn_temp,p);
%         end
%     end
    
    AgentDist=AgentDistnext;
end

% GeneralEqmCondnPath=GeneralEqmCondnPath;

WeightedSumSq_GeneralEqmCondnPath=sum(sum(transpathoptions.weightsforpath.*(GeneralEqmCondnPath).^2));

WeightedSumSq_GeneralEqmCondnPath=gather(WeightedSumSq_GeneralEqmCondnPath);

if transpathoptions.verbose==1
    fprintf('Current PricePath: \n')
    PricePathOld
    fprintf('Current GeneralEqmCondnPath: \n')
    GeneralEqmCondnPath
    fprintf('Current WeightedSumSq_GeneralEqmCondnPath: \n')
    WeightedSumSq_GeneralEqmCondnPath
end

if transpathoptions.graphpricepath==1
    figure(transpathoptions.pricepathfig);
    plot(PricePathOld)
%     label(PricePathNames)
end


end