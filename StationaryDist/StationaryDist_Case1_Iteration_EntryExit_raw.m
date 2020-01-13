function [StationaryDistKron]=StationaryDist_Case1_Iteration_EntryExit_raw(StationaryDistKron,Parameters,EntryExitParamNames,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1, and then keep
% track of actual mass using StationaryDistKron.mass.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Get the Entry-Exit parameters out of Parameters.
CondlProbOfSurvival=Parameters.(EntryExitParamNames.CondlProbOfSurvival{1});
DistOfNewAgents=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
MassOfNewAgents=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
% StationaryDistKron.mass
% StationaryDistKron.pdf

% simoptions.DistOfNewAgents=kron(pistar_tau,pistar_s); % Note: these should be in 'reverse order'
% simoptions.CondlProbOfSurvivalParamNames={'lambda'};
% simoptions.CondlProbOfSurvival=@(lambda) 1-lambda;
% simoptions.MassOfNewAgents

%% Get the entry and exit variables into the appropriate form.
% % Check whether CondlProbOfSurvival is a function, matrix, or scalar, and act accordingly.
% if isa(simoptions.CondlProbOfSurvival,'function_handle')
%     % Implicitly assume there is only one parameter in simoptions.CondlProbOfSurvivalParamNames
%     Values=StateDependentParam_az(Params,simoptions.CondlProbOfSurvivalParamNames{:},DependenceVec,n_a,n_z,1,1,simoptions.parallel);
%     CondlProbOfSurvival=arrayfun(simoptions.CondlProbOfSurvival, Values);
%     CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
% Check whether CondlProbOfSurvival is a matrix, or scalar, and act accordingly.
if isscalar(CondlProbOfSurvival)
    % No need to do anything
elseif isa(CondlProbOfSurvival,'numeric')
    CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
else % Does not appear to have been inputted correctly
    fprintf('ERROR: CondlProbOfSurvival parameter does not appear to have been inputted with correct format')
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

%% First, create Ptranspose
% First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
% (Actually I create it's transpose, as that is what will be used repeatedly later.)

if N_d==0 %length(n_d)==1 && n_d(1)==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
end
if simoptions.endogenousexit==1
    optaprime=optaprime+(1-CondlProbOfSurvival'); % endogenous exit means that CondlProbOfSurvival will be 1-ExitPolicy
    % This will make all those who 'exit' instead move to first point on
    % 'grid on a'. Since as part of it's creation Ptranspose then gets multiplied by the
    % CondlProbOfSurvival these agents will all 'die' anyway.
    % It is done as otherwise the optaprime policy is being stored as
    % 'zero' for those who exit, and this causes an error when trying to
    % use optaprime as an index.
    % (Need to use transpose of CondlProbOfSurvival because it is being
    % kept in the 'transposed' form as usually is used to multiply Ptranspose.)
end


if simoptions.parallel<2
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
%     else
%         optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
%     end
    Ptranspose=zeros(N_a,N_a*N_z);
    Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
    else
%         size(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z]))
%         size(Ptranspose)
%         size((ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)
        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z 
    end
elseif simoptions.parallel==2 % Using the GPU
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
%     else
%         optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
%     end
    Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
    Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
        Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose));
    else
        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z 
    end
elseif simoptions.parallel>2
%     if N_d==0 %length(n_d)==1 && n_d(1)==0
%         optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
%     else
%         optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
%     end
    Ptranspose=sparse(N_a,N_a*N_z);
    Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
    if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
    else
        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z 
    end
end

%% The rest is essentially the same regardless of which simoption.parallel is being used
%SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z); % This line was handy when checking/debugging. Have left it here.
if simoptions.parallel==2
    SteadyStateDistKronOld=zeros(N_a*N_z,1,'gpuArray');
else
    SteadyStateDistKronOld=zeros(N_a*N_z,1);
end
SScurrdist=sum(abs(StationaryDistKron.pdf-SteadyStateDistKronOld));
SScounter=0;

% Switch into 'mass times pdf' form, and work with that until get
% convergence, then switch solution back into seperate mass and pdf form
% for output.
StationaryDistKron.pdf=StationaryDistKron.mass*StationaryDistKron.pdf; % Make it the pdf

while SScurrdist>simoptions.tolerance && (100*SScounter)<simoptions.maxit

%     StationaryDistKron.pdf=StationaryDistKron.mass*StationaryDistKron.pdf; % Actually work with mass*pdf for next few lines. Makes adding in the new agents and removing the dying much easier.
    for jj=1:100
       %% Following line is essentially the only change that entry and exit require to the actual iteration
        % Note that it works with cdf, rather than pdf. So there are also
        % some lines just pre and post to do that.
%         StationaryDistKron.pdf=(MassOfNewAgents/StationaryDistKron.mass)*DistOfNewAgentsKron+Ptranspose*(CondlProbOfSurvival.*StationaryDistKron.pdf); %No point checking distance every single iteration. Do 100, then check.
        StationaryDistKron.pdf=MassOfNewAgents*DistOfNewAgentsKron+Ptranspose*StationaryDistKron.pdf; %No point checking distance every single iteration. Do 100, then check.
        % Note: Exit, captured in the CondlProbOfSurvival is already included into Ptranspose when it is created.
%         StationaryDistKron.mass=sum(sum(StationaryDistKron.pdf));
%         StationaryDistKron.pdf=StationaryDistKron.pdf./StationaryDistKron.mass; % Make it the pdf
%         StationaryDistKron.mass=MassOfNewAgents+sum(CondlProbOfSurvival.*StationaryDistKron.pdf)*StationaryDistKron.mass;
    end
%     StationaryDistKron.mass=sum(sum(StationaryDistKron.pdf));
%     StationaryDistKron.pdf=StationaryDistKron.pdf/StationaryDistKron.mass; % Make it the pdf

    SteadyStateDistKronOld=StationaryDistKron.pdf;
%     StationaryDistKron.pdf=Ptranspose*StationaryDistKron.pdf; % Base the tolerance on 10 iterations. (For some reason just using one iteration worked perfect on gpu, but was not accurate enough on cpu)
    StationaryDistKron.pdf=MassOfNewAgents*DistOfNewAgentsKron+Ptranspose*StationaryDistKron.pdf; %No point checking distance every single iteration. Do 100, then check.
    SScurrdist=sum(abs(StationaryDistKron.pdf-SteadyStateDistKronOld));
    % Note: I just look for convergence in the pdf and 'assume' the mass
    % will also have converged by then. I should probably correct this.
    
    SScounter=SScounter+1;
    if simoptions.verbose==1
        if rem(SScounter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', SScounter, SScurrdist, simoptions.tolerance)            
        end
    end
end

% Turn it into the 'mass and pdf' format required for output.
StationaryDistKron.mass=sum(sum(StationaryDistKron.pdf));
StationaryDistKron.pdf=StationaryDistKron.pdf/StationaryDistKron.mass; % Make it the pdf


if simoptions.parallel>=3 % Solve with sparse matrix
    StationaryDistKron.pdf=full(StationaryDistKron.pdf);
    if simoptions.parallel==4 % Solve with sparse matrix, but return answer on gpu.
        StationaryDistKron.pdf=gpuArray(StationaryDistKron.pdf);
    end
end

if ~((100*SScounter)<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
