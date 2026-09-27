function StationaryDistKron=StationaryDist_InfHorz_Iteration_EntryExit2_raw(StationaryDistKron,Policy_aprime,N_a,N_z,pi_z, ExitProb, EntryDist, simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Note that EntryDist is of size N_a*N_z-by-1.

% First, get Gamma
optaprime=gather(reshape(Policy_aprime,[1,N_a*N_z]));
ExitProb=gather(ExitProb);
pi_z=sparse(gather(pi_z));

Gammatranspose=sparse(optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a)),1:1:N_a*N_z,(1-ExitProb)*ones(N_a*N_z,1),N_a*N_z,N_a*N_z);


%% The rest is essentially the same regardless of which simoption.parallel is being used
% StationaryDistKron=sparse(N_a*N_z,1);
StationaryDistKron=sparse(gather(StationaryDistKron));

currdist=Inf;
counter=0;
currdistprev=Inf; nocontraction=0; % for the non-convergence guard below
while currdist>simoptions.tolerance && (100*counter)<simoptions.maxit

    for jj=1:100
        % Tan improvement
        StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
        StationaryDistKron=StationaryDistKron+ExitProb*EntryDist;
    end
    StationaryDistKronOld=StationaryDistKron;

    % Tan improvement
    StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);
    StationaryDistKron=StationaryDistKron+ExitProb*EntryDist;

    currdist=sum(abs(StationaryDistKron-StationaryDistKronOld));

    % Non-convergence guard. With entry and exit the distribution is NOT normalised inside this
    % loop -- StationaryDistKron is the unnormalised measure whose total is the agent mass -- so
    % currdist is the L1 change in MASS. If exit is unreachable at these parameters, mass simply
    % accumulates at MassOfNewAgents per period, currdist sits flat, and there is no stationary
    % distribution to find. Left alone that grinds the whole simoptions.maxit budget (default 10^6
    % iterations). A contraction factor of essentially 1 sustained over 1000 periods means the answer
    % is not going to arrive: stop and say so, rather than burning 10^8 matrix products.
    % Deliberately conservative (0.9999, ten consecutive blocks): a merely slow model still runs.
    % full() because the distribution is held sparse here, so currdist is a sparse scalar and
    % warning() refuses sparse inputs (the non-entry-exit raw has the same note on its while).
    contractionfactor=full(currdist/currdistprev);
    if contractionfactor>0.9999
        nocontraction=nocontraction+1;
    else
        nocontraction=0;
    end
    if nocontraction>=10
        warning('VFIToolkit:StationaryDistEntryExitNotContracting', ...
            ['StationaryDist with entry-exit is not contracting: factor %.6f per 100 periods after %i periods. ' ...
             'The agent mass is not converging, which usually means exit is unreachable at these parameters, ' ...
             'so no stationary distribution exists. Stopping early and returning the last iterate.'], ...
            contractionfactor,100*counter)
        break
    end
    currdistprev=currdist;

    counter=counter+1;
    if simoptions.verbose==1
        if rem(counter,50)==0
            % full() because the distribution is held sparse here, so currdist is a sparse scalar and
            % fprintf refuses sparse inputs. This print had never been exercised on the entry-exit path.
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', 100*counter, full(currdist), simoptions.tolerance)
        end
    end
end

if simoptions.parallel==2
    StationaryDistKron=gpuArray(full(StationaryDistKron));
else
    StationaryDistKron=full(StationaryDistKron);
end

if ~((100*counter)<simoptions.maxit) % 100 applications per counter, so iterations is 100*counter
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end




end
