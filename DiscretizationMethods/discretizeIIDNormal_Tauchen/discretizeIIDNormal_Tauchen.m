function [e_grid,pi_e]=discretizeIIDNormal_Tauchen(mew,sigma,enum,Tauchen_q, tauchenoptions)
% Create states vector, e_grid, and probability vector, pi_e, for the discrete approximation
%    of iid process e~N(mew,sigma^2), by Tauchen method
%
% Inputs
%   mew            - mean
%   sigma          - standard deviation
%   enum           - number of states in discretization of e (must be an odd number)
%   Tauchen_q      - (Hyperparameter) Defines max/min grid points as mew+-Tauchen_q*sigma (I suggest 2 or 3)
%                    Set Tauchen_q=[] to use the default of min(sqrt(enum-1),4), which is the same
%                    rule discretizeAR1_Tauchen uses; see the note at that default below.
% Optional Inputs (tauchenoptions)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
%   e_grid:        - (enum-by-1) skip grid construction and put the probabilities on this grid.
%                    Tauchen_q is ignored. Bin edges are the midpoints between adjacent grid
%                    points, and the two outermost bins extend to -Inf and +Inf, so the grid need
%                    not be evenly spaced and pi_e sums to one whatever grid is given.
% Outputs
%   e_grid         - column vector containing the enum states of the discrete approximation of e
%   pi_e           - column vector of probabilities of the discrete approximation of e;
%                    pi_e(i) is the probability of state i (sums to 1)
%
% Note: this is the normal-distribution-only command. It was created as a copy of
% discretizeIID_Tauchen(), which is to be generalized to handle a wide range of
% distributions; this command will only ever handle the normal distribution.
%%%%%%%%%%%%%%%
% Original paper:
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"

if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
end

% Tauchen_q=[] means use the default width, the same rule as discretizeAR1_Tauchen. There the
% reasoning is that sqrt(enum-1) is the half-width the Rouwenhorst construction requires and
% discretizeAR1_FarmerToda takes as its default, while Tauchen pays for extra width in grid spacing
% and so in the variance - past the truncation regime its variance error is bin-width smearing,
% h^2/12 with h the spacing - so the width that minimises the error settles rather than keeps
% growing, and the cap at 4 tracks where it settles. None of that is weaker for an iid process: it
% is the same grid construction and the same smearing, just without the conditional distribution.
%
% It is ignored when tauchenoptions.e_grid is set, since then no grid is being constructed.
if isempty(Tauchen_q)
    Tauchen_q=min(sqrt(enum-1),4);
end

% A user grid is signalled by its presence, exactly as in discretizeAR1_Tauchen
if isfield(tauchenoptions,'e_grid')
    tauchenoptions.usergrid=1;
    if size(tauchenoptions.e_grid,2)>1
        tauchenoptions.e_grid=tauchenoptions.e_grid'; % use a column internally
    end
    if length(tauchenoptions.e_grid)~=enum
        error('length of tauchenoptions.e_grid must equal enum')
    end
    if ~issorted(tauchenoptions.e_grid,'strictascend')
        error('tauchenoptions.e_grid must be strictly ascending')
    end
else
    tauchenoptions.usergrid=0;
end

if enum==1
    e_grid=mew; %expected value of e
    pi_e=1;
    if tauchenoptions.parallel==2
        e_grid=gpuArray(e_grid);
        pi_e=gpuArray(pi_e);
    end
    return
end

if tauchenoptions.parallel==0 || tauchenoptions.parallel==1
    if tauchenoptions.usergrid==1
        e_grid=tauchenoptions.e_grid;
        edges=(e_grid(1:end-1)+e_grid(2:end))/2;
        upper=[edges; e_grid(end)]; % last entry overwritten below (extends to +inf)
        lower=[e_grid(1); edges];   % first entry overwritten below (extends to -inf)
    else
        e_grid=mew*ones(enum,1) + linspace(-Tauchen_q*sigma,Tauchen_q*sigma,enum)';
        omega=e_grid(2)-e_grid(1); % all the points are equidistant by construction
        upper=e_grid+omega/2;
        lower=e_grid-omega/2;
    end

    P_part1=0.5*erfc(-(upper-mew)./(sigma*sqrt(2)));
    P_part2=0.5*erfc(-(lower-mew)./(sigma*sqrt(2)));

    pi_e=P_part1-P_part2;
    pi_e(1)=P_part1(1);
    pi_e(enum)=1-P_part2(enum);

elseif tauchenoptions.parallel==2 %Parallelize on GPU
    if tauchenoptions.usergrid==1
        e_grid=gpuArray(tauchenoptions.e_grid);
        edges=(e_grid(1:end-1)+e_grid(2:end))/2;
        upper=[edges; e_grid(end)]; % last entry overwritten below (extends to +inf)
        lower=[e_grid(1); edges];   % first entry overwritten below (extends to -inf)
    else
        e_grid=gpuArray(mew*ones(enum,1) + linspace(-Tauchen_q*sigma,Tauchen_q*sigma,enum)');
        omega=e_grid(2)-e_grid(1); % all the points are equidistant by construction
        upper=e_grid+omega/2;
        lower=e_grid-omega/2;
    end

    % Same erfc expression as the cpu branch above, so the two differ only where the cpu and gpu
    % erfc libraries disagree in the last bit. erfc not 1+erf: the left tail cdf is tiny, and
    % 1+erf loses all relative precision there (it is exactly zero past about -8.3 sd).

    erfcinput=arrayfun(@(ei,mew,sigma) -(ei-mew)/(sigma*sqrt(2)), upper,mew,sigma);
    P_part1=0.5*erfc(erfcinput);

    erfcinput=arrayfun(@(ei,mew,sigma) -(ei-mew)/(sigma*sqrt(2)), lower,mew,sigma);
    P_part2=0.5*erfc(erfcinput);

    pi_e=P_part1-P_part2;
    pi_e(1)=P_part1(1);
    pi_e(enum)=1-P_part2(enum);

end

end
