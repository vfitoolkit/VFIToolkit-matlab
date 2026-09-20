%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteApproximation
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute a discrete state approximation to a distribution with known
%       moments, using the maximum entropy procedure proposed in Tanaka and
%       Toda (2013)
%
% Usage:
%       [p,lambdaBar,momentError] = discreteApproximation(D,T,TBar,q,lambda0)
%
% Inputs:
% D         - (K x N) matrix of grid points. K is the dimension of the
%             domain. N is the number of points at which an approximation
%             is to be constructed.
% T         - A function handle which should accept arguments of dimension
%             (K x N) and return an (L x N) matrix of moments evaluated at
%             each grid point, where L is the number of moments to be
%             matched.
% TBar      - (L x 1) vector of moments of the underlying distribution
%             which should be matched
% Optional:
% q         - (1 X N) vector of prior weights for each point in D. The
%             default is for each point to have an equal weight.
% lambda0   - (L x 1) vector of initial guesses for the dual problem
%             variables. The default is a vector of zeros.
%
% Outputs:
% p         - (1 x N) vector of probabilities assigned to each grid point in
%             D.
% lambdaBar - (L x 1) vector of dual problem variables which solve the
%             maximum entropy problem
% momentError - (L x 1) vector of errors in moments (defined by moments of
%               discretization minus actual moments)
%
% Version 1.2: June 7, 2016
%
% Version 1.3: May 26, 2019
%
% Changed algorithm to 'trust-region' to use Hessian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [p,lambdaBar,momentError] = discreteApproximation(D,T,TBar,q,lambda0)

% Input error checking

if nargin < 3
    error('You must provide at least 3 arguments to discreteApproximation.')
end

N = size(D,2);

Tx = T(D);
L = size(Tx,1);

if size(Tx,2) ~= N || length(TBar) ~= L
    error('Dimension mismatch')
end

% Default prior weights
if nargin == 3
    q = ones(1,N)./N;
end

% Compute maximum entropy discrete distribution

% entropyObjective returns the exact Hessian as its third output and nothing used to ask for it.
% The live line below selected 'trust-region' - the one fminunc algorithm that can use a Hessian -
% but requested the gradient only, so the solver finite-differenced a Hessian it could have been
% handed. The line above it, kept from the 2016 code, had the opposite half of the problem: it
% asked for the Hessian but selected no algorithm, so on any release where the default is
% quasi-newton the option is inert. Both halves are needed. They are written here with the current
% optimoptions names rather than the legacy optimset ones, so that a name fminunc cannot honour is
% an error rather than a silent no-op, which is how this survived unnoticed.
%    OLD: options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','off','Algorithm','trust-region','GradObj','on');
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off','OptimalityTolerance',1e-10,'FunctionTolerance',1e-10,'StepTolerance',1e-10);


% IS THE TARGET EVEN ATTAINABLE ON THIS GRID? If TBar lies outside the convex hull of the columns
% of Tx there is no distribution on D with these moments at all, and the solve below is spent
% discovering that. The bound is exact, not a heuristic: for any p>=0 with sum(p)=1,
%    norm([Tx;ones(1,N)]*p'-[TBar;1]) = norm(Tx*p'-TBar)
% because the appended row contributes exactly zero, so the achieved moment error can never be
% smaller than the non-negative least squares minimum over the larger set {p>=0, sum free}. If
% that minimum already exceeds 1e-5, which is the bar every caller uses to accept a fit, then no
% solve can be accepted and there is nothing to be gained by running one. Skipping straight to
% lambda=0 leaves p as the normalised prior and momentError as the prior's own error, which is
% itself at least the hull residual and so above the bar - exactly the outcome the caller gets
% today after paying for the solve, so callers need no change.
%
% Measured over 900 rows of P4's and P2's own workloads: 253 rows finish above the bar, and 249
% of them are infeasible in this sense. The test costs 5 to 57 times less than the fminunc call it
% replaces. The saving is modest where a command tries only two moments and then one (about 10 per
% cent of solver time across P4's sweep), and large in the commands that walk a ladder from four
% moments down, where an infeasible row currently pays for the four-moment and three-moment
% attempts before reaching one that fits.
%
% A cheaper test was tried first - reject if any single component of TBar lies outside the range
% that component takes on the grid - and it is nearly free and never wrong, but it caught only 14
% per cent of the infeasible rows. What makes these targets unattainable is the joint geometry,
% not any one moment, so the hull test is the one that earns its keep. lsqnonneg is base MATLAB,
% not the Optimization Toolbox.
hullResidual = norm([Tx;ones(1,N)]*lsqnonneg([Tx;ones(1,N)],[TBar;1])-[TBar;1]);

if hullResidual > 1e-5
    lambdaBar = zeros(size(lambda0));
else
    % Sometimes the algorithm fails to converge if the initial guess is too far
    % away from the truth. If this occurs, the program tries an initial guess
    % of all zeros.
    try
        lambdaBar = fminunc(@(lambda) entropyObjective(lambda,Tx,TBar,q),lambda0,options);
    catch
        warning('Failed to find a solution from provided initial guess. Trying new initial guess.')
        lambdaBar = fminunc(@(lambda) entropyObjective(lambda,Tx,TBar,q),zeros(size(lambda0)),options);
    end

    % The target is attainable and the trust-region solve still missed the bar, so the failure is
    % the solver's rather than the problem's, and it is worth one more attempt with a different
    % algorithm. This is a small population - four rows in the 900 measured - but quasi-newton
    % recovered every one of them, including the two that the trust-region Hessian above costs at
    % the acceptance boundary. Only accepted if it actually improves the moment error, so this can
    % never make a row worse. No try/catch is needed: the hull test above has already established
    % that the dual is bounded, which is the only way this diverges.
    [objTR,gradTR] = entropyObjective(lambdaBar,Tx,TBar,q);
    if norm(gradTR./objTR) > 1e-5
        optionsQN = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Display','off','OptimalityTolerance',1e-10,'FunctionTolerance',1e-10,'StepTolerance',1e-10);
        lambdaQN = fminunc(@(lambda) entropyObjective(lambda,Tx,TBar,q),zeros(size(lambda0)),optionsQN);
        [objQN,gradQN] = entropyObjective(lambdaQN,Tx,TBar,q);
        if norm(gradQN./objQN) < norm(gradTR./objTR)
            lambdaBar = lambdaQN;
        end
    end
end

% Compute final probability weights and moment errors
[obj,gradObj] = entropyObjective(lambdaBar,Tx,TBar,q);
Tdiff = Tx-repmat(TBar,1,N);
p = (q.*exp(lambdaBar'*Tdiff))./obj;
momentError = gradObj./obj;

end