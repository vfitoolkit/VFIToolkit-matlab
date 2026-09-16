function [e_grid,pi_e,otheroutputs] = discretizeIIDNormal_TanakaToda(mew,sigma,enum,tanakatodaoptions)
% Please cite: Tanaka & Toda (2013) - "Discrete approximations of continuous distributions by maximum entropy"
%
% Create states vector, e_grid, and probability vector, pi_e, for the discrete approximation
%    of iid process e~N(mew,sigma^2), by Tanaka-Toda (maximum entropy) method.
%
% Inputs
%   mew            - mean
%   sigma          - standard deviation
%   enum           - number of states in discretization of e (minimum of 3)
% Optional Inputs (tanakatodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%   nMoments       - Number of moments to match (default=2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigma (default depends on enum)
%   e_grid         - (enum-by-1) skip grid construction and run the maximum entropy step on this
%                    grid. method and nSigmas are ignored. The prior is treated as in 'even', which
%                    is the right translation because 'even' is the one built-in grid that carries no
%                    quadrature weights of its own (W=ones); the other three pair a specific weight
%                    vector with a specific set of nodes, and those weights are meaningless on a grid
%                    the caller chose.
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   e_grid         - column vector containing the enum states of the discrete approximation of e
%   pi_e           - column vector of probabilities of the discrete approximation of e;
%   otheroutputs   - optional output structure containing info for evaluating the discretization including,
%        otheroutputs.nMoments  - how many moments the maximum entropy problem actually matched.
%              This can be fewer than tanakatodaoptions.nMoments, as the solve falls back a moment
%              at a time when it fails. There is no loop here, so this is a scalar rather than a grid.
%                    pi_e(i) is the probability of state i (sums to 1)
%
% Note: this is the normal-distribution-only command. It was created as a copy of
% discretizeIID_TanakaToda(), which is to be generalized to handle a wide range of
% distributions; this command will only ever handle the normal distribution.
%
% This code is modified from that of Toda & Farmer (v: https://github.com/alexisakira/discretization
% Please cite them if you use this.
% This version was lightly modified by Robert Kirkby
%
%%%%%%%%%%%%%%%
% Original paper:
% Tanaka & Toda (2013) - "Discrete approximations of continuous distributions by maximum entropy"
% This is the iid analogue of the Farmer-Toda (2017) method for AR(1).

%% Set defaults
if ~exist('tanakatodaoptions','var')
    tanakatodaoptions.nMoments=2;
    tanakatodaoptions.nSigmas = min(sqrt(2*(enum-1)),3);
    tanakatodaoptions.method='gauss-hermite';
    tanakatodaoptions.parallel=1+(gpuDeviceCount>0);
    tanakatodaoptions.verbose=1;
else
    if ~isfield(tanakatodaoptions,'nMoments')
        tanakatodaoptions.nMoments = 2; % Default number of moments to match is 2
    end
    if ~isfield(tanakatodaoptions,'nSigmas')
        tanakatodaoptions.nSigmas = min(sqrt(2*(enum-1)),3);
    end
    if ~isfield(tanakatodaoptions,'method')
        tanakatodaoptions.method='gauss-hermite';
    end
    if ~isfield(tanakatodaoptions,'parallel')
        tanakatodaoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(tanakatodaoptions,'verbose')
        tanakatodaoptions.verbose=1;
    end
end

%% Check inputs are correctly formatted
if ~isnumeric(enum) || enum < 3 || rem(enum,1) ~= 0
    error('enum must be a positive integer greater than 3')
end

if ~isnumeric(tanakatodaoptions.nMoments) || tanakatodaoptions.nMoments < 1 || tanakatodaoptions.nMoments > 4 || ~((rem(tanakatodaoptions.nMoments,1) == 0) || (tanakatodaoptions.nMoments == 1))
    error('tanakatodaoptions.nMoments must be either 1, 2, 3, 4')
end

% A user grid is signalled by its presence, exactly as in discretizeAR1_Tauchen and
% discretizeIIDNormal_Tauchen. nSigmas is not used in that case, so its warning must not fire either.
if isfield(tanakatodaoptions,'e_grid')
    tanakatodaoptions.usergrid=1;
    if size(tanakatodaoptions.e_grid,2)>1
        tanakatodaoptions.e_grid=tanakatodaoptions.e_grid'; % use a column until the switch below, which works in rows
    end
    if length(tanakatodaoptions.e_grid)~=enum
        error('length of tanakatodaoptions.e_grid must equal enum')
    end
    if ~issorted(tanakatodaoptions.e_grid,'strictascend')
        error('tanakatodaoptions.e_grid must be strictly ascending')
    end
else
    tanakatodaoptions.usergrid=0;
    if tanakatodaoptions.nSigmas<1.2
        warning('Trying to hit the 2nd moment with tanakatodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for tanakatodaoptions.nSigmas<1.2).')
    end
end

if tanakatodaoptions.usergrid==1
    e_grid = tanakatodaoptions.e_grid'; % the code below works in row vectors
    W = ones(1,enum);
else
    switch tanakatodaoptions.method
        case 'even'
            e_grid = linspace(mew-tanakatodaoptions.nSigmas*sigma,mew+tanakatodaoptions.nSigmas*sigma,enum);
            W = ones(1,enum);
        case 'gauss-legendre'
            [e_grid,W] = legpts(enum,[mew-tanakatodaoptions.nSigmas*sigma,mew+tanakatodaoptions.nSigmas*sigma]);
            e_grid = e_grid';
        case 'clenshaw-curtis'
            [e_grid,W] = fclencurt(enum,mew-tanakatodaoptions.nSigmas*sigma,mew+tanakatodaoptions.nSigmas*sigma);
            e_grid = fliplr(e_grid');
            W = fliplr(W');
        case 'gauss-hermite'
            [e_grid,W] = GaussHermite(enum);
            e_grid = mew+sqrt(2)*sigma*e_grid';
            W = W'./sqrt(pi);
    end
end

%% define central moments that Tanaka-Toda method targets
T1 = 0;
T2 = sigma^2;
T3 = 0;
T4 = 3*sigma^4;

TBar = [T1 T2 T3 T4]'; % vector of central moments


%% Tanaka-Toda method
scalingFactor = max(abs(e_grid));
kappa = 1e-8;

if strcmp(tanakatodaoptions.method,'gauss-hermite') && tanakatodaoptions.usergrid==0  % define prior probabilities
    q = W;
else
    q = W.*(exp(-0.5*((e_grid-mew)./sigma).^2)./(sigma*sqrt(2*pi)));
end

if any(q < kappa)
    q(q < kappa) = kappa; % replace by small number for numerical stability
end

nMoments_matched=0; % Used to record how many moments the maximum entropy solve actually hit
if tanakatodaoptions.nMoments == 1 % match only 1 moment
    pi_e = discreteApproximation(e_grid,@(x)(x-mew)/scalingFactor,TBar(1)./scalingFactor,q,0);
    nMoments_matched=1;
else % match 2 moments first
    [p,lambda,momentError] = discreteApproximation(e_grid,@(x) [(x-mew)./scalingFactor;...
        ((x-mew)./scalingFactor).^2],...
        TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
    if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
        if tanakatodaoptions.verbose==1
            warning('Failed to match first 2 moments. Just matching 1.')
        end
        pi_e = discreteApproximation(e_grid,@(x)(x-mew)/scalingFactor,0,q,0);
        nMoments_matched=1;
    elseif tanakatodaoptions.nMoments == 2
        pi_e = p;
        nMoments_matched=2;
    elseif tanakatodaoptions.nMoments == 3 % 3 moments
        [pnew,~,momentError] = discreteApproximation(e_grid,@(x) [(x-mew)./scalingFactor;...
            ((x-mew)./scalingFactor).^2;((x-mew)./scalingFactor).^3],...
            TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
        if norm(momentError) > 1e-5
            if tanakatodaoptions.verbose==1
                warning('Failed to match first 3 moments.  Just matching 2.')
            end
            pi_e = p;
            nMoments_matched=2;
        else
            pi_e = pnew;
            nMoments_matched=3;
        end
    elseif tanakatodaoptions.nMoments == 4 % 4 moments
        [pnew,~,momentError] = discreteApproximation(e_grid,@(x) [(x-mew)./scalingFactor;...
            ((x-mew)./scalingFactor).^2; ((x-mew)./scalingFactor).^3;...
            ((x-mew)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
        if norm(momentError) > 1e-5
            [pnew,~,momentError] = discreteApproximation(e_grid,@(x) [(x-mew)./scalingFactor;...
                ((x-mew)./scalingFactor).^2;((x-mew)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError) > 1e-5
                if tanakatodaoptions.verbose==1
                    warning('Failed to match first 3 moments.  Just matching 2.')
                end
                pi_e = p;
                nMoments_matched=2;
            else
                pi_e = pnew;
                nMoments_matched=3;
                if tanakatodaoptions.verbose==1
                    warning('Failed to match first 4 moments.  Just matching 3.')
                end
            end
        else
            pi_e = pnew;
            nMoments_matched=4;
        end
    end
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments=nMoments_matched; % How many moments the maximum entropy problem actually matched

if tanakatodaoptions.parallel==2
    e_grid=gpuArray(e_grid);
    pi_e=gpuArray(pi_e);
end

e_grid=e_grid'; % Output as column vector
pi_e=pi_e'; % Output as column vector

end
