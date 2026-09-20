function [e_grid,pi_e,otheroutputs] = discretizeIID_TanakaToda(mew,sigma,enum,tanakatodaoptions)
% Please cite: Tanaka & Toda (2013) - "Discrete approximations of continuous distributions by maximum entropy"
%
% Create states vector, e_grid, and probability vector, pi_e, for the discrete approximation
%    of an iid process, by the Tanaka-Toda (maximum entropy) method. The default is e~N(mew,sigma^2),
%    and the options below generalise it to other distributions, to truncations of them, to a
%    caller-supplied grid, to moments given directly, and to distributions with mass points.
%
% Inputs
%   mew            - mean (of the normal; unused and may be [] for other distributions, and for the
%                    normal itself when targetmoments is given)
%   sigma          - standard deviation (same: unused and may be [] whenever the distribution is not
%                    the normal, or targetmoments supplies the moments directly)
%   enum           - number of states in discretization of e (minimum of 3)
% Optional Inputs (tanakatodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%                    Default 'gauss-hermite' for the normal, 'even' otherwise, and 'even' whenever
%                    targetmoments is given: the Gauss-Hermite nodes and weights are built for a
%                    gaussian and mean nothing for anything else. 'gauss-hermite' is also REJECTED
%                    with a finite truncate - its nodes are unbounded by construction, so there is
%                    no way to keep them inside the truncation interval.
%   nMoments       - Number of moments to match (default=2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigma (default depends on enum)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
%   distribution   - 'normal' (default), 'uniform', 'exponential' or 'lognormal'. Parameters for the
%                    non-normal families go in distparams, NOT in mew/sigma:
%                       'uniform'      distparams.lb, distparams.ub
%                       'exponential'  distparams.lambda      (support [0,Inf))
%                       'lognormal'    distparams.mu, distparams.sigma, of the UNDERLYING normal
%   distparams     - struct of parameters for the chosen distribution, as above
%   truncate       - [lb,ub], restrict the distribution to this interval and renormalise. Either
%                    bound may be -Inf or Inf for one-sided truncation; an infinite bound cannot
%                    set a grid edge, so that side falls back to the mean +- nSigmas*(std dev).
%   e_grid         - (enum-by-1 or 1-by-enum) put the probabilities on this grid rather than
%                    constructing one. method and nSigmas are then ignored.
%   targetmoments  - [mean, variance, 3rd central moment, 4th central moment]. NOTE the first entry
%                    is the MEAN, not zero: the moment functions are centred on it, exactly as mew is
%                    used in the default path, so both pieces of information are needed. Overrides
%                    distribution and truncate for the TARGETS only.
%   prior          - the maximum entropy prior. Either a function handle (a density, evaluated at the
%                    grid points) or a vector of length enum. Overrides the density that would
%                    otherwise be used.
%   masspoints     - (m-by-2) [value, probability] rows, for a distribution with atoms. The moment
%                    matching then describes the CONTINUOUS part - the distribution conditional on
%                    not landing on an atom - and the atoms are added afterwards.
% Outputs
%   e_grid         - column vector containing the states of the discrete approximation of e. This is
%                    enum states, except when masspoints adds atoms that do not coincide with an
%                    existing grid point, in which case it is longer.
%   pi_e           - column vector of probabilities of the discrete approximation of e;
%                    pi_e(i) is the probability of state i (sums to 1)
%   otheroutputs   - optional output structure containing info for evaluating the discretization including,
%        otheroutputs.nMoments  - how many moments the maximum entropy problem actually matched.
%              This can be fewer than tanakatodaoptions.nMoments, as the solve falls back a moment
%              at a time when it fails. There is no loop here, so this is a scalar rather than a grid.
%              IT IS WORTH READING. A grid that is too wide for its number of nodes fails silently
%              and falls back on the prior, which can put the standard deviation out by a factor of
%              four while every row still sums to one.
%        otheroutputs.momentError - the norm of the moment error at the ACCEPTED solution. When the
%              ladder falls back this is the error of the fit that was KEPT, not of the one that was
%              rejected on the way - so momentError<1e-05 holds whenever nMoments>=2, which is the
%              threshold the ladder accepts at.
%        otheroutputs.TBar      - the central moments actually targeted, so the caller can check
%                                 what was asked for
%        otheroutputs.q         - the maximum entropy prior actually used
%        otheroutputs.mew_eff   - the point the moment functions were centred on
%
% PRECEDENCE, since three things get chosen and each needs an unambiguous order:
%   grid    - e_grid if given; else the truncate interval via method; else mew+-nSigmas*sigma via method
%   targets - targetmoments if given; else distribution combined with truncate; else the normal closed form
%   prior   - prior if given; else the quadrature weights alone when targetmoments is given; else
%             the density (truncated if applicable); else the quadrature weights alone.
%             The targetmoments rule does NOT depend on whether a distribution was also named:
%             targetmoments overrides distribution for the targets, so it overrides it for the prior.
%
% Every option above defaults off, and discretizeIID_TanakaToda(mew,sigma,enum) with any of the four
% method values is bit-for-bit what it was before they existed. discretizeIIDNormal_TanakaToda is a
% frozen copy of that default path and the test bank compares the two, so a change to it shows up.
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
% WHAT THE CALLER ACTUALLY SET is recorded before anything is defaulted in, because two decisions
% below depend on it: the method default differs by distribution, and a caller-supplied grid makes
% method and nSigmas unused, which is worth saying rather than silently dropping them.
if ~exist('tanakatodaoptions','var')
    tanakatodaoptions=struct();
end
usersetmethod=isfield(tanakatodaoptions,'method');
usersetnSigmas=isfield(tanakatodaoptions,'nSigmas');
if ~isfield(tanakatodaoptions,'distribution')
    tanakatodaoptions.distribution='normal';
end
if ~isfield(tanakatodaoptions,'method')
    if isfield(tanakatodaoptions,'targetmoments') && ~isempty(tanakatodaoptions.targetmoments)
        % Moments given directly: there is no gaussian for Gauss-Hermite to place its nodes for, and
        % mew/sigma may legitimately be empty, so the default has to be the interval method.
        tanakatodaoptions.method='even';
    elseif strcmp(tanakatodaoptions.distribution,'normal')
        tanakatodaoptions.method='gauss-hermite'; % unchanged from before these options existed
    else
        tanakatodaoptions.method='even';
    end
end
if ~isfield(tanakatodaoptions,'nMoments')
    tanakatodaoptions.nMoments = 2; % Default number of moments to match is 2
end
if ~isfield(tanakatodaoptions,'nSigmas')
    tanakatodaoptions.nSigmas = min(sqrt(2*(enum-1)),3);
end
if ~isfield(tanakatodaoptions,'parallel')
    tanakatodaoptions.parallel=1+(gpuDeviceCount>0);
end
if ~isfield(tanakatodaoptions,'verbose')
    tanakatodaoptions.verbose=1;
end
% The generalising options, all defaulting to off
if ~isfield(tanakatodaoptions,'distparams')
    tanakatodaoptions.distparams=struct();
end
if ~isfield(tanakatodaoptions,'truncate')
    tanakatodaoptions.truncate=[];
end
if ~isfield(tanakatodaoptions,'targetmoments')
    tanakatodaoptions.targetmoments=[];
end
if ~isfield(tanakatodaoptions,'prior')
    tanakatodaoptions.prior=[];
end
if ~isfield(tanakatodaoptions,'masspoints')
    tanakatodaoptions.masspoints=[];
end
% A caller-supplied grid is signalled by its presence, as in discretizeAR1_Tauchen and its siblings
if isfield(tanakatodaoptions,'e_grid')
    tanakatodaoptions.usergrid=1;
    % Must be on the cpu: the entropy problem is solved with fminunc(), which cannot take gpuArrays.
    tanakatodaoptions.e_grid=gather(tanakatodaoptions.e_grid);
    if size(tanakatodaoptions.e_grid,1)>1
        tanakatodaoptions.e_grid=tanakatodaoptions.e_grid'; % use a row internally
    end
    if length(tanakatodaoptions.e_grid)~=enum
        error('length of tanakatodaoptions.e_grid must equal enum')
    end
    if ~issorted(tanakatodaoptions.e_grid,'strictascend')
        error('tanakatodaoptions.e_grid must be strictly ascending')
    end
    % method and nSigmas are not used when a grid is given. Say so rather than silently dropping
    % them, and skip the nSigmas warning below, which would fire on a quantity no longer in use.
    if tanakatodaoptions.verbose==1 && (usersetmethod || usersetnSigmas)
        warning('tanakatodaoptions.e_grid was given, so tanakatodaoptions.method and tanakatodaoptions.nSigmas are ignored')
    end
else
    tanakatodaoptions.usergrid=0;
end

%% Check inputs are correctly formatted
if ~isnumeric(enum) || enum < 3 || rem(enum,1) ~= 0
    error('enum must be a positive integer greater than 3')
end

if ~isnumeric(tanakatodaoptions.nMoments) || tanakatodaoptions.nMoments < 1 || tanakatodaoptions.nMoments > 4 || ~((rem(tanakatodaoptions.nMoments,1) == 0) || (tanakatodaoptions.nMoments == 1))
    error('tanakatodaoptions.nMoments must be either 1, 2, 3, 4')
end

% nSigmas is not in use when a grid was supplied, so the warning must not fire on it then.
if tanakatodaoptions.nSigmas<1.2 && tanakatodaoptions.usergrid==0
    warning('Trying to hit the 2nd moment with tanakatodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for tanakatodaoptions.nSigmas<1.2).')
end

%% Validate the generalising options
if ~any(strcmp(tanakatodaoptions.distribution,{'normal','uniform','exponential','lognormal'}))
    error('tanakatodaoptions.distribution must be one of normal, uniform, exponential, lognormal')
end
% mew and sigma are documented as unused for the non-normal families, and they genuinely are - but
% for the normal with no targetmoments they are the whole specification, so an empty one has to say
% so here rather than surface later as a dimension error from inside a grid construction.
if strcmp(tanakatodaoptions.distribution,'normal') && isempty(tanakatodaoptions.targetmoments)
    if isempty(mew) || isempty(sigma)
        error('mew and sigma cannot be empty for the normal distribution unless tanakatodaoptions.targetmoments is given: they are what specifies it')
    end
    if sigma<=0
        error('sigma must be strictly positive')
    end
end
if ~isempty(tanakatodaoptions.truncate)
    if length(tanakatodaoptions.truncate)~=2 || tanakatodaoptions.truncate(1)>=tanakatodaoptions.truncate(2)
        error('tanakatodaoptions.truncate must be [lb,ub] with lb<ub (either may be infinite)')
    end
end
if ~isempty(tanakatodaoptions.targetmoments)
    if length(tanakatodaoptions.targetmoments)~=4
        error('tanakatodaoptions.targetmoments must be a 4-vector [mean, variance, 3rd central moment, 4th central moment]')
    end
    if tanakatodaoptions.targetmoments(2)<=0
        error('tanakatodaoptions.targetmoments(2) is the variance and must be strictly positive')
    end
end
if ~isempty(tanakatodaoptions.masspoints)
    if size(tanakatodaoptions.masspoints,2)~=2
        error('tanakatodaoptions.masspoints must be an m-by-2 matrix of [value,probability] rows')
    end
    if any(tanakatodaoptions.masspoints(:,2)<=0) || sum(tanakatodaoptions.masspoints(:,2))>1+10^(-12)
        error('tanakatodaoptions.masspoints probabilities must be strictly positive and sum to at most one')
    end
end
if ~strcmp(tanakatodaoptions.distribution,'normal') && strcmp(tanakatodaoptions.method,'gauss-hermite') && tanakatodaoptions.usergrid==0
    error('tanakatodaoptions.method=''gauss-hermite'' places its nodes and weights for a gaussian, so it cannot be used with tanakatodaoptions.distribution=''%s''',tanakatodaoptions.distribution)
end


%% Resolve the distribution: the density, the natural support, and the target moments
% THE DEFAULT PATH IS KEPT IN CLOSED FORM. For an untruncated normal the moments are written down
% rather than integrated, so discretizeIID_TanakaToda(mew,sigma,enum) is bit-for-bit what it was
% before any of this existed. Every other case goes through one quadrature path, which composes
% automatically with truncation instead of needing a truncated closed form per family.
switch tanakatodaoptions.distribution
    case 'normal'
        pdfh=@(x) exp(-0.5*((x-mew)./sigma).^2)./(sigma*sqrt(2*pi));
        natlo=-Inf; nathi=Inf;
    case 'uniform'
        if ~isfield(tanakatodaoptions.distparams,'lb') || ~isfield(tanakatodaoptions.distparams,'ub')
            error('tanakatodaoptions.distribution=''uniform'' needs distparams.lb and distparams.ub')
        end
        ulb=tanakatodaoptions.distparams.lb; uub=tanakatodaoptions.distparams.ub;
        if ulb>=uub
            error('tanakatodaoptions.distparams.lb must be less than distparams.ub')
        end
        pdfh=@(x) ((x>=ulb)&(x<=uub))/(uub-ulb);
        natlo=ulb; nathi=uub;
    case 'exponential'
        if ~isfield(tanakatodaoptions.distparams,'lambda')
            error('tanakatodaoptions.distribution=''exponential'' needs distparams.lambda')
        end
        elam=tanakatodaoptions.distparams.lambda;
        if elam<=0
            error('tanakatodaoptions.distparams.lambda must be strictly positive')
        end
        pdfh=@(x) elam*exp(-elam*x).*(x>=0);
        natlo=0; nathi=Inf;
    case 'lognormal'
        if ~isfield(tanakatodaoptions.distparams,'mu') || ~isfield(tanakatodaoptions.distparams,'sigma')
            error('tanakatodaoptions.distribution=''lognormal'' needs distparams.mu and distparams.sigma, of the UNDERLYING normal')
        end
        lmu=tanakatodaoptions.distparams.mu; lsd=tanakatodaoptions.distparams.sigma;
        if lsd<=0
            error('tanakatodaoptions.distparams.sigma must be strictly positive')
        end
        pdfh=@(x) (x>0).*exp(-0.5*((log(max(x,realmin))-lmu)./lsd).^2)./(max(x,realmin)*lsd*sqrt(2*pi));
        natlo=0; nathi=Inf;
end

% Truncation narrows the support and renormalises the density. An infinite bound leaves that side
% as it was, which is the one place truncate and nSigmas interact - see the grid bounds below.
trunlo=natlo; trunhi=nathi;
if ~isempty(tanakatodaoptions.truncate)
    trunlo=max(natlo,tanakatodaoptions.truncate(1));
    trunhi=min(nathi,tanakatodaoptions.truncate(2));
    if trunlo>=trunhi
        error('tanakatodaoptions.truncate leaves no mass: the interval does not overlap the support of the distribution')
    end
end
istruncated=~isempty(tanakatodaoptions.truncate) && (trunlo>natlo || trunhi<nathi);
% GAUSS-HERMITE CANNOT RESPECT A FINITE TRUNCATION. Its nodes are the roots of a Hermite polynomial,
% scaled about the mean - they are unbounded by construction and take no interval argument, so there
% is no way to keep them inside [lb,ub]. Before this guard the branch simply ignored the truncation:
% with truncate=[-1,1] at enum=9 the grid spanned [-4.51,4.51] and eight of the nine nodes sat
% outside the interval, carrying 27% of the probability at points where the truncated density is
% exactly zero. The moments still came out right, because the entropy fit compensates - which is
% what made it silent. Same reasoning as the gauss-hermite guard for non-normal families above.
if istruncated && strcmp(tanakatodaoptions.method,'gauss-hermite') && tanakatodaoptions.usergrid==0
    error('tanakatodaoptions.method=''gauss-hermite'' cannot be used with a finite tanakatodaoptions.truncate: its nodes are unbounded by construction and cannot be kept inside the truncation interval')
end
if istruncated
    Ztrunc=integral(pdfh,trunlo,trunhi,'AbsTol',1e-14,'RelTol',1e-12);
    pdfh_eff=@(x) pdfh(x).*(x>=trunlo).*(x<=trunhi)/Ztrunc;
else
    pdfh_eff=pdfh;
end

% The targets. targetmoments wins outright; otherwise the closed form for an untruncated normal,
% and quadrature on the density for everything else.
if ~isempty(tanakatodaoptions.targetmoments)
    tm=tanakatodaoptions.targetmoments;
    mew_eff=tm(1); % the MEAN, which is what the moment functions are centred on
    TBar=[0; tm(2); tm(3); tm(4)];
    distsd=sqrt(tm(2));
elseif strcmp(tanakatodaoptions.distribution,'normal') && ~istruncated
    mew_eff=mew;
    TBar=[0; sigma^2; 0; 3*sigma^4]; % the closed form this command has always used
    distsd=sigma;
else
    rawm=zeros(4,1);
    for k_c=1:4
        rawm(k_c)=integral(@(x) (x.^k_c).*pdfh_eff(x),trunlo,trunhi,'AbsTol',1e-14,'RelTol',1e-12);
    end
    mew_eff=rawm(1);
    c2=rawm(2)-mew_eff^2;
    c3=rawm(3)-3*mew_eff*rawm(2)+2*mew_eff^3;
    c4=rawm(4)-4*mew_eff*rawm(3)+6*mew_eff^2*rawm(2)-3*mew_eff^4;
    TBar=[0; c2; c3; c4];
    distsd=sqrt(c2);
end

%% The grid
% Bounds: a finite truncation or natural support sets them; an infinite side falls back to nSigmas
% standard deviations from the mean, which is what the normal has always done.
if isfinite(trunlo)
    gridlo=trunlo;
else
    gridlo=mew_eff-tanakatodaoptions.nSigmas*distsd;
end
if isfinite(trunhi)
    gridhi=trunhi;
else
    gridhi=mew_eff+tanakatodaoptions.nSigmas*distsd;
end

if tanakatodaoptions.usergrid==1
    e_grid=tanakatodaoptions.e_grid; % row vector
    W=ones(1,enum); % treat like 'even', the one method whose weights are not tied to its own nodes
else
    switch tanakatodaoptions.method
        case 'even'
            e_grid = linspace(gridlo,gridhi,enum);
            W = ones(1,enum);
        case 'gauss-legendre'
            [e_grid,W] = legpts(enum,[gridlo,gridhi]);
            e_grid = e_grid';
        case 'clenshaw-curtis'
            [e_grid,W] = fclencurt(enum,gridlo,gridhi);
            e_grid = fliplr(e_grid');
            W = fliplr(W');
        case 'gauss-hermite'
            % mew_eff and distsd, NOT mew and sigma. On the default path they are the same values by
            % literal assignment, so this is bit-for-bit what it was; everywhere else mew and sigma
            % may be empty or may not describe the distribution being discretized at all.
            [e_grid,W] = GaussHermite(enum);
            e_grid = mew_eff+sqrt(2)*distsd*e_grid';
            W = W'./sqrt(pi);
    end
end

%% Tanaka-Toda method
scalingFactor = max(abs(e_grid));
kappa = 1e-8;

% THE PRIOR, in the stated precedence order: an explicit prior wins; else the density, truncated if
% applicable; else the quadrature weights alone. Gauss-Hermite weights already encode a gaussian
% density, so multiplying by the density again would double-count it - hence the third branch.
if ~isempty(tanakatodaoptions.prior)
    if isa(tanakatodaoptions.prior,'function_handle')
        q = W.*tanakatodaoptions.prior(e_grid);
    else
        q = reshape(gather(tanakatodaoptions.prior),1,[]);
        if length(q)~=enum
            error('tanakatodaoptions.prior, given as a vector, must have enum entries')
        end
    end
elseif strcmp(tanakatodaoptions.method,'gauss-hermite') && tanakatodaoptions.usergrid==0
    q = W;
elseif ~isempty(tanakatodaoptions.targetmoments)
    % TARGETMOMENTS GIVEN: the prior is the weights alone, whether or not a distribution was also
    % named. targetmoments already overrides distribution for the targets, so overriding it for the
    % prior too is the consistent rule and the easier one to document. The previous version made this
    % conditional on the caller NOT having named a distribution, which meant it worked if you said
    % nothing and broke if you said 'normal' - the opposite way round from anyone's expectation, and
    % it reached pdfh_eff's closure over an empty mew and sigma.
    q = W;
else
    q = W.*pdfh_eff(e_grid);
end

if any(q < kappa)
    q(q < kappa) = kappa; % replace by small number for numerical stability
end

nMoments_matched=0; % Used to record how many moments the maximum entropy solve actually hit
% EACH SOLVE KEEPS ITS OWN ERROR. momentError used to be one reused name, so when the ladder fell
% back - accepting the two-moment fit after three moments failed - it reported the error of the
% REJECTED three-moment solve. A caller testing momentError<1e-5 would read a perfectly good
% two-moment fit as a blow-up, which is the exact hazard otheroutputs was added to remove. The
% one-moment branches also used to assert momErrOut=0 rather than capture what the solve returned.
momErrOut=NaN; % the moment error at whichever solution is accepted below
if tanakatodaoptions.nMoments == 1 % match only 1 moment
    [pi_e,~,momentError1] = discreteApproximation(e_grid,@(x)(x-mew_eff)/scalingFactor,TBar(1)./scalingFactor,q,0);
    nMoments_matched=1; momErrOut=norm(momentError1);
else % match 2 moments first
    [p,lambda,momentError2] = discreteApproximation(e_grid,@(x) [(x-mew_eff)./scalingFactor;...
        ((x-mew_eff)./scalingFactor).^2],...
        TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
    if norm(momentError2) > 1e-5 % if 2 moments fail, then just match 1 moment
        if tanakatodaoptions.verbose==1
            warning('Failed to match first 2 moments. Just matching 1.')
        end
        [pi_e,~,momentError1] = discreteApproximation(e_grid,@(x)(x-mew_eff)/scalingFactor,0,q,0);
        nMoments_matched=1; momErrOut=norm(momentError1);
    elseif tanakatodaoptions.nMoments == 2
        pi_e = p;
        nMoments_matched=2; momErrOut=norm(momentError2);
    elseif tanakatodaoptions.nMoments == 3 % 3 moments
        [pnew,~,momentError3] = discreteApproximation(e_grid,@(x) [(x-mew_eff)./scalingFactor;...
            ((x-mew_eff)./scalingFactor).^2;((x-mew_eff)./scalingFactor).^3],...
            TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
        if norm(momentError3) > 1e-5
            if tanakatodaoptions.verbose==1
                warning('Failed to match first 3 moments.  Just matching 2.')
            end
            pi_e = p;
            nMoments_matched=2; momErrOut=norm(momentError2); % the ACCEPTED fit, not the rejected one
        else
            pi_e = pnew;
            nMoments_matched=3; momErrOut=norm(momentError3);
        end
    elseif tanakatodaoptions.nMoments == 4 % 4 moments
        [pnew,~,momentError4] = discreteApproximation(e_grid,@(x) [(x-mew_eff)./scalingFactor;...
            ((x-mew_eff)./scalingFactor).^2; ((x-mew_eff)./scalingFactor).^3;...
            ((x-mew_eff)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
        if norm(momentError4) > 1e-5
            [pnew,~,momentError3] = discreteApproximation(e_grid,@(x) [(x-mew_eff)./scalingFactor;...
                ((x-mew_eff)./scalingFactor).^2;((x-mew_eff)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError3) > 1e-5
                if tanakatodaoptions.verbose==1
                    warning('Failed to match first 3 moments.  Just matching 2.')
                end
                pi_e = p;
                nMoments_matched=2; momErrOut=norm(momentError2); % the ACCEPTED fit
            else
                pi_e = pnew;
                nMoments_matched=3; momErrOut=norm(momentError3);
                if tanakatodaoptions.verbose==1
                    warning('Failed to match first 4 moments.  Just matching 3.')
                end
            end
        else
            pi_e = pnew;
            nMoments_matched=4; momErrOut=norm(momentError4);
        end
    end
end

%% Mass points
% The moment matching above describes the CONTINUOUS part - the distribution conditional on not
% landing on an atom - so it is scaled by the probability of that, and the atoms are added on top.
%
% ONE CONSEQUENCE IS COUNTERINTUITIVE AND IS NOT A BUG. The probability that ends up on the node at
% an atom's value is NOT the atom's probability: the continuous part also puts mass on that node,
% and the grid cannot tell the two apart. For the zeta shock of Guvenen-McKay-Ryan Model 2 the atom
% is 0.560 but the node ends at 0.714, because the continuous part has mass just below zero. The
% mass belongs near zero either way, so this is correct - but it looks wrong if you expected 0.560.
if ~isempty(tanakatodaoptions.masspoints)
    mp=tanakatodaoptions.masspoints;
    pmass=sum(mp(:,2));
    pi_e=(1-pmass)*pi_e;
    nodetol=1e-12*max(1,max(abs(e_grid)));
    for k_c=1:size(mp,1)
        [dmin,imin]=min(abs(e_grid-mp(k_c,1)));
        if dmin<=nodetol % the atom sits on an existing node, so merge into it
            pi_e(imin)=pi_e(imin)+mp(k_c,2);
        else % otherwise it becomes a new node
            e_grid=[e_grid,mp(k_c,1)];
            pi_e=[pi_e,mp(k_c,2)];
        end
    end
    [e_grid,ord]=sort(e_grid);
    pi_e=pi_e(ord);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments=nMoments_matched; % How many moments the maximum entropy problem actually matched
otheroutputs.momentError=momErrOut;     % the norm at the ACCEPTED solution, not at the last one tried
otheroutputs.TBar=TBar;                 % the central moments actually targeted
otheroutputs.q=q;                       % the maximum entropy prior actually used
otheroutputs.mew_eff=mew_eff;           % the point the moment functions were centred on

if tanakatodaoptions.parallel==2
    e_grid=gpuArray(e_grid);
    pi_e=gpuArray(pi_e);
end

e_grid=e_grid'; % Output as column vector
pi_e=pi_e'; % Output as column vector

end
