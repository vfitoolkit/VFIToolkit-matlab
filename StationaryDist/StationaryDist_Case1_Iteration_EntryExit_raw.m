function [StationaryDistKron]=StationaryDist_Case1_Iteration_EntryExit_raw(StationaryDistKron,Parameters,EntryExitParamNames,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1, and then keep
% track of actual mass using StationaryDistKron.mass.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Get the Entry-Exit parameters out of Parameters.
CondlProbOfSurvival=Parameters.(EntryExitParamNames.CondlProbOfSurvival{1});
DistOfNewAgents=sparse(gather(Parameters.(EntryExitParamNames.DistOfNewAgents{1})));
MassOfNewAgents=gather(Parameters.(EntryExitParamNames.MassOfNewAgents{1}));
% StationaryDistKron.mass
% StationaryDistKron.pdf

% simoptions.DistOfNewAgents=kron(pistar_tau,pistar_s); % Note: these should be in 'reverse order'
% simoptions.CondlProbOfSurvivalParamNames={'lambda'};
% simoptions.CondlProbOfSurvival=@(lambda) 1-lambda;
% simoptions.MassOfNewAgents

%% Get the entry and exit variables into the appropriate form.
% Check whether CondlProbOfSurvival is a matrix, or scalar, and act accordingly.
if isscalar(gather(CondlProbOfSurvival))
    CondlProbOfSurvival=gather(CondlProbOfSurvival);
elseif isa(gather(CondlProbOfSurvival),'numeric')
    CondlProbOfSurvival=gather(reshape(CondlProbOfSurvival,[1,N_a*N_z]));
else % Does not appear to have been inputted correctly
    fprintf('ERROR: CondlProbOfSurvival parameter does not appear to have been inputted with correct format \n')
    dbstack
    return
end
% Move these to where they need to be.
DistOfNewAgentsKron=sparse(gather(reshape(DistOfNewAgents,[N_a*N_z,1])));

%% First, set up for Tan improvement (create Gammatranspose, and make pi_z sparse)

if N_d==0
    optaprime=gather(reshape(PolicyIndexesKron,[1,N_a*N_z]));
else
    optaprime=gather(reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]));
end

if simoptions.endogenousexit==0
    Gammatranspose=sparse(optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a)),1:1:N_a*N_z,CondlProbOfSurvival.*ones(N_a*N_z,1),N_a*N_z,N_a*N_z);
elseif simoptions.endogenousexit==1
    % Note: the (optaprime>0) handles the endogenous exit decisions (the decision to exit is optaprime=0)
    II1=optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a));
    II2=1:1:N_a*N_z;
    VV=CondlProbOfSurvival.*ones(N_a*N_z,1);
    Gammatranspose=sparse(II1(optaprime>0),II2(optaprime>0),VV(optaprime>0),N_a*N_z,N_a*N_z);
elseif simoptions.endogenousexit==2
    exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
    exitprobs=[1-sum(exitprobabilities),exitprobabilities];
    % Mixed exit (endogenous and exogenous), so we know that CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
    Gammatranspose=sparse(optaprime++kron(N_a*(0:1:N_z-1),ones(1,N_a)),1:1:N_a*N_z,(exitprobs(1)+exitprobs(2)*CondlProbOfSurvival).*ones(N_a*N_z,1),N_a*N_z,N_a*N_z);

    % NOTE TO SELF: This wasn't tested when I converted to Tan improvement (as is not in any of the three firm
    % models implemented in toolkit), so following is a copy-paste backup
    % of how it worked without Tan improvement
    % Ptranspose=sparse(N_a,N_a*N_z);
    % Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    % % Ptranspose1=(kron(pi_z',ones(N_a,N_a))).*(kron(exitprob(1)*ones(N_z,1),Ptranspose)); % No exit, and remove exog exit
    % % Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
    % % Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
    % % Following line does (in one line) what the above three commented out lines do (doing it in one presumably reduces memory usage of Ptranspose1 and Ptranspose2)
    % Ptranspose=((kron(pi_z',ones(N_a,N_a))).*(kron(exitprobs(1)*ones(N_z,1),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
end
pi_z_sparse=sparse(gather(pi_z));


%% The rest is essentially the same regardless of which simoption.parallel is being used
StationaryDistKronOld=sparse(N_a*N_z,1);
currdist=sum(abs(StationaryDistKron.pdf-StationaryDistKronOld));
counter=0;

% Switch into 'mass times pdf' form, and work with that until get
% convergence, then switch solution back into seperate mass and pdf form for output.
StationaryDistKron_pdf=sparse(gather(StationaryDistKron.mass*StationaryDistKron.pdf)); % Make it the pdf

% simoptions

while currdist>simoptions.tolerance && counter<simoptions.maxit

    for jj=1:100
       %% Following line is essentially the only change that entry and exit require to the actual iteration
        % Note that it works with cdf, rather than pdf. So there are also some lines just pre and post to do that.

        % Two steps of the Tan improvement
        StationaryDistKron_pdf=reshape(Gammatranspose*StationaryDistKron_pdf,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        StationaryDistKron_pdf=reshape(StationaryDistKron_pdf*pi_z_sparse,[N_a*N_z,1]);

        StationaryDistKron_pdf=MassOfNewAgents*DistOfNewAgentsKron+StationaryDistKron_pdf; %No point checking distance every single iteration. Do 100, then check.
        % Note: Exit, captured in the CondlProbOfSurvival is already included into Ptranspose when it is created.
    end
    StationaryDistKronOld=StationaryDistKron_pdf;

    % Two steps of the Tan improvement
    StationaryDistKron_pdf=reshape(Gammatranspose*StationaryDistKron_pdf,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    StationaryDistKron_pdf=reshape(StationaryDistKron_pdf*pi_z_sparse,[N_a*N_z,1]);

    StationaryDistKron_pdf=MassOfNewAgents*DistOfNewAgentsKron+StationaryDistKron_pdf; %No point checking distance every single iteration. Do 100, then check.
    
    currdist=sum(abs(StationaryDistKron_pdf-StationaryDistKronOld));
    % Note: I just look for convergence in the pdf and 'assume' the mass will also have converged by then. I should probably correct this.
    
    counter=counter+1;
    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', counter, currdist, simoptions.tolerance)            
        end
    end
end
counter

if simoptions.parallel==2
    StationaryDistKron.pdf=gpuArray(full(StationaryDistKron_pdf));
else
    StationaryDistKron.pdf=full(StationaryDistKron_pdf);
end
% Turn it into the 'mass and pdf' format required for output.
StationaryDistKron.mass=sum(sum(StationaryDistKron.pdf));
StationaryDistKron.pdf=StationaryDistKron.pdf/StationaryDistKron.mass; % Make it the pdf

if ~((100*counter)<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
