function [z_grid_J,pi_z_J,jequaloneDistz,otheroutputs]=discretizeLifeCycleAR1wGM_Tauchen(mew,rho,mixprobs_i,mu_i,sigma_i,znum,J,Tauchen_q,tauchenoptions)
% Tauchen method for a life-cycle (age-dependent) AR(1) process with gaussian mixture innovations:
%       z(j) = mew(j) + rho(j)*z(j-1) + e(j),   e(j) ~ F(j)
%          where F(j) = sum_{i=1}^nmix mixprobs_i(i,j)*N(mu_i(i,j),sigma_i(i,j)^2)
%       with initial condition z(0)=0 by default
%
% This is the age-dependent extension of discretizeAR1wGM_Tauchen, in the same way that
% discretizeLifeCycleAR1_FellaGallipoliPanTauchen is the age-dependent extension of
% discretizeAR1_Tauchen. It is the Tauchen counterpart of discretizeLifeCycleAR1wGM_KFTT.
%
% Note: nmix, the number of components in the mixture, cannot depend on j.
%
% Inputs:
%   mew          - (1-by-J) vector of intercepts (NOT the mean of z; see the note on centring below)
%   rho          - (1-by-J) vector of autocorrelation coefficients
%   mixprobs_i   - (nmix-by-J) mixture probabilities (each column must sum to one)
%   mu_i         - (nmix-by-J) means of the mixture components
%   sigma_i      - (nmix-by-J) standard deviations of the mixture components
%   znum         - number of grid points (the same at every age)
%   J            - number of ages
%   Tauchen_q    - (Hyperparameter) the grid at age j is E(z_j) +- Tauchen_q*sd(z_j).
%                  Can be a scalar, or a vector with one element per age.
%                  Set Tauchen_q=[] for the default, min(sqrt(znum-1),w), with w the width that
%                  leaves the same tail mass outside the grid as four standard deviations does
%                  for a normal. w is computed per age, so the default is an age vector when the
%                  mixture varies with age. See the note where it is computed below.
% Optional inputs (tauchenoptions)
%   parallel     - set equal to 2 to return the outputs as gpuArrays
%   verbose      - set to 0 to silence the recommendation printed below
%        The initial period can be controlled with the following. By default z(0)=0.
%   initialj0mewz    - give period 0 a mean of z0 (a point mass, unless initialj0sigmaz is also set)
%   initialj0sigmaz  - make period 0 a N(z0,initialj0sigmaz^2)
%        Or set period 1 directly, instead of period 0:
%   initialj1mewz    - period 1 has mean initialj1mewz
%   initialj1sigmaz  - period 1 is N(initialj1mewz,initialj1sigmaz^2).
%                      Give initialj1sigmaz as a VECTOR to make period 1 a gaussian mixture, in
%                      which case initialj1mixprobs and initialj1mu must be given as vectors of
%                      the same length. (Same convention as discretizeLifeCycleAR1wGM_KFTT.)
% Outputs:
%   z_grid_J       - znum-by-J matrix, column j is the grid at age j
%   pi_z_J         - znum-by-znum-by-(J-1) matrix; pi_z_J(:,:,j) is the transition from age j to
%                    age j+1. There are only J-1 of them, as there is no age J+1 to transition to.
%   jequaloneDistz - znum-by-1 distribution of z at age 1
%   otheroutputs   - structure with
%        otheroutputs.sigma_z        - (1-by-J) standard deviation of z at each age
%        otheroutputs.mew_z          - (1-by-J) mean of z at each age
%        otheroutputs.jequalzeroDistz - (znum-by-1) the period 0 distribution, when period 0 is used
%
% ON GRID CENTRING. The grid at age j is centred on E(z_j), which includes the mean of the mixture:
% the recursion is mewz(j) = mew(j) + E(e_j) + rho(j)*mewz(j-1). This matches discretizeAR1wGM_Tauchen.
% discretizeLifeCycleAR1wGM_KFTT instead centres on a recursion with no E(e) term, which coincides
% only when the mixture is mean zero. The two conventions disagree for a non-mean-zero mixture, and
% that is an open question rather than an accident: the KFTT command carries a
% setmixturemutoenforcezeromean option, which suggests the design intent was that mixtures be mean
% zero. P7 of the DiscretizationMethodTests test bank measures the gap directly, on a calibration
% chosen to be non-mean-zero precisely so that it can.
%
% ON THE METHOD. Conditional on z at age j, the value at age j+1 is a gaussian mixture with the same
% weights and standard deviations as F(j+1) but with every component mean shifted by the conditional
% mean mew(j+1)+rho(j+1)*z. So the probability of landing in a bin of the age j+1 grid is the
% mixture-weighted sum of normal cdf differences over that bin, one term per component. Everything
% is computed on the cpu and moved to the gpu at the end if asked for, which is what
% discretizeLifeCycleAR1_FellaGallipoliPanTauchen does.
%%%%%
% Original paper (for the Tauchen method itself, which is for gaussian innovations):
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"

%% Set options
if ~exist('tauchenoptions','var')
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
    tauchenoptions.verbose=1;
else
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(tauchenoptions,'verbose')
        tauchenoptions.verbose=1;
    end
end

% The recommendation below is printed once per call, and the test bank that exercises these
% commands calls them hundreds of times, so it is gated. Set tauchenoptions.verbose=0 to silence it.
if tauchenoptions.verbose==1
    fprintf('COMMENT: The Tauchen method is likely inferior to the KFTT method for discretizing life-cycle AR(1) processes with gaussian mixture innovations. \n')
    fprintf('         It is suggested you consider using discretizeLifeCycleAR1wGM_KFTT instead. \n')
end


%% Check the inputs
if znum<2
    error('The state space must have znum>1')
end
nmix=size(mixprobs_i,1);
if size(mu_i,1)~=nmix || size(sigma_i,1)~=nmix
    error('mixprobs_i, mu_i and sigma_i must all have the same number of rows (nmix)')
end
if size(mixprobs_i,2)~=J || size(mu_i,2)~=J || size(sigma_i,2)~=J
    error('mixprobs_i, mu_i and sigma_i must all have J columns (one per age)')
end
if isscalar(mew)
    mew=mew*ones(1,J); % a scalar is taken to be an age-independent parameter
end
if isscalar(rho)
    rho=rho*ones(1,J);
end
if length(mew)~=J || length(rho)~=J
    error('mew and rho must be scalars, or vectors with one element per age (J)')
end
mew=reshape(mew,[1,J]); rho=reshape(rho,[1,J]);
if any(mixprobs_i(:)<0)
    error('mixture probabilities (mixprobs_i) must be non-negative')
end
if max(abs(sum(mixprobs_i,1)-1))>10^(-12)
    error('mixture probabilities (mixprobs_i) must sum to one at every age')
end
if any(sigma_i(:)<=0)
    error('standard deviations of the mixture components (sigma_i) must be strictly positive')
end

%% Moments of the mixture innovations at each age
mew_e=sum(mixprobs_i.*mu_i,1);                              % (1-by-J) mean of e(j)
sigmasq_e=sum(mixprobs_i.*(mu_i.^2+sigma_i.^2),1)-mew_e.^2; % (1-by-J) variance of e(j)

% Tauchen_q=[] means use the default width
% For a gaussian innovation the default is min(sqrt(znum-1),4), as in discretizeAR1_Tauchen. A cap
% of 4 is wrong for a gaussian mixture, though, and the test bank measures how wrong: on P3's
% calibration the width that minimises the excess kurtosis error is 7, not 4. What sets the
% requirement is how far the CONDITIONAL distribution reaches, since that is what each row of the
% transition matrix is built from - not the unconditional kurtosis of z, which fails as a predictor
% (P3 and P4 have nearly the same kurtosis of z, 0.68 and 0.83, and want widths a factor of two
% apart, because P4's conditional law is a single normal while P3's is a mixture).
%
% So the cap is computed rather than fixed: pick the width that leaves the same tail mass outside
% the grid as 4 standard deviations does for a normal, which for a gaussian mixture is closed form.
% The only tuned number is the 4, inherited from the gaussian default, so a one-component mixture
% returns exactly 4 and nothing changes for a normal. On P3's mixture it returns 7.01 against a
% measured optimum of 7, which was not fitted.
%
% Solved by bisection; the tail is monotone decreasing in the width, so this always converges.
% The mixture may differ by age, so the computed width is an age vector - which Tauchen_q already
% accepts. This block sits below the moments because it needs them.
if isempty(Tauchen_q)
    epstail=2*(1-(0.5*erfc(-4/sqrt(2)))); % the mass a normal leaves outside four standard deviations
    Tauchen_q=zeros(1,J);
    for jj=1:J
        sigma_ej=sqrt(sigmasq_e(jj));
        wlo=0.5; whi=40;
        for bisect_c=1:200
            wmid=(wlo+whi)/2;
            tailmass=sum(mixprobs_i(:,jj).*((1-(0.5*erfc(-((mew_e(jj)+wmid*sigma_ej-mu_i(:,jj))./sigma_i(:,jj))/sqrt(2))))+(0.5*erfc(-((mew_e(jj)-wmid*sigma_ej-mu_i(:,jj))./sigma_i(:,jj))/sqrt(2)))));
            if tailmass>epstail
                wlo=wmid;
            else
                whi=wmid;
            end
        end
        Tauchen_q(jj)=min(sqrt(znum-1),(wlo+whi)/2);
    end
end
if isscalar(Tauchen_q)
    Tauchen_q=Tauchen_q*ones(1,J);
end
if length(Tauchen_q)~=J
    error('Tauchen_q must be a scalar, or a vector with one element per age (J)')
end
Tauchen_q=reshape(Tauchen_q,[1,J]);

%% Period 0, and hence period 1
z0=0;
if isfield(tauchenoptions,'initialj0mewz')
    z0=tauchenoptions.initialj0mewz;
end
if isfield(tauchenoptions,'initialj0sigmaz')
    sigmaz0=tauchenoptions.initialj0sigmaz;
else
    sigmaz0=0;
end

mewz=zeros(1,J); sigmaz=zeros(1,J);
% Period 1 from period 0. z(1)=mew(1)+rho(1)*z(0)+e(1), and z(0) is independent of e(1), so the
% variance adds and the mean is the mean of the mixture plus the shifted period 0 mean.
mewz(1)=mew(1)+rho(1)*z0+mew_e(1);
sigmaz(1)=sqrt(rho(1)^2*sigmaz0^2+sigmasq_e(1));

% Period 1 set directly, which overwrites the above
j1ismixture=0;
if isfield(tauchenoptions,'initialj1sigmaz') && ~isscalar(tauchenoptions.initialj1sigmaz)
    % period 1 is itself a gaussian mixture
    j1ismixture=1;
    if ~isfield(tauchenoptions,'initialj1mixprobs') || ~isfield(tauchenoptions,'initialj1mu')
        error('a vector initialj1sigmaz means period 1 is a gaussian mixture, so initialj1mixprobs and initialj1mu must be given as vectors of the same length')
    end
    j1probs=reshape(tauchenoptions.initialj1mixprobs,[],1);
    j1mu=reshape(tauchenoptions.initialj1mu,[],1);
    j1sigma=reshape(tauchenoptions.initialj1sigmaz,[],1);
    if length(j1probs)~=length(j1mu) || length(j1probs)~=length(j1sigma)
        error('initialj1mixprobs, initialj1mu and initialj1sigmaz must all be the same length')
    end
    if abs(sum(j1probs)-1)>10^(-12)
        error('initialj1mixprobs must sum to one')
    end
    mewz(1)=sum(j1probs.*j1mu);
    sigmaz(1)=sqrt(sum(j1probs.*(j1mu.^2+j1sigma.^2))-mewz(1)^2);
elseif isfield(tauchenoptions,'initialj1mewz') && isfield(tauchenoptions,'initialj1sigmaz')
    mewz(1)=tauchenoptions.initialj1mewz;
    sigmaz(1)=tauchenoptions.initialj1sigmaz;
elseif isfield(tauchenoptions,'initialj1mewz')
    if ~isscalar(tauchenoptions.initialj1mewz)
        error('a vector initialj1mewz without a vector initialj1sigmaz would be a normal distribution with a vector mean, which is not a thing')
    end
    mewz(1)=tauchenoptions.initialj1mewz;
    sigmaz(1)=0;
elseif isfield(tauchenoptions,'initialj1sigmaz')
    mewz(1)=0;
    sigmaz(1)=tauchenoptions.initialj1sigmaz;
end

%% The rest of the age profile
for jj=2:J
    mewz(jj)=mew(jj)+mew_e(jj)+rho(jj)*mewz(jj-1);
    sigmaz(jj)=sqrt(rho(jj)^2*sigmaz(jj-1)^2+sigmasq_e(jj));
end

%% Step 1: the grid at each age
z_grid_J=zeros(znum,J);
for jj=1:J
    if sigmaz(jj)>0
        z_grid_J(:,jj)=linspace(mewz(jj)-Tauchen_q(jj)*sigmaz(jj),mewz(jj)+Tauchen_q(jj)*sigmaz(jj),znum)';
    else
        % a degenerate age: the process is a point mass, so spread the grid slightly around it so
        % the grid is still strictly ascending, and put all the mass on the middle point below
        z_grid_J(:,jj)=mewz(jj)+linspace(-10^(-6),10^(-6),znum)';
    end
end

%% Step 2: the transition matrices, from age jj to age jj+1
% pi_z_J(:,:,jj) is determined by the age jj+1 parameters, hence the jj+1 indexing.
pi_z_J=zeros(znum,znum,J-1);
for jj=1:J-1
    zlag=z_grid_J(:,jj);
    znext=z_grid_J(:,jj+1);
    omega=znext(2)-znext(1); % the age jj+1 grid is evenly spaced by construction
    upper=znext+omega/2;
    lower=znext-omega/2;
    upperj=ones(znum,1)*upper';
    lowerj=ones(znum,1)*lower';
    zi=zlag*ones(1,znum);
    P_part1=zeros(znum,znum);
    P_part2=zeros(znum,znum);
    for i_c=1:nmix
        % the conditional mean of the i-th component, given z at age jj
        P_part1=P_part1+mixprobs_i(i_c,jj+1)*(0.5*erfc(-((upperj-rho(jj+1)*zi-mu_i(i_c,jj+1))-mew(jj+1))./(sigma_i(i_c,jj+1)*sqrt(2))));
        P_part2=P_part2+mixprobs_i(i_c,jj+1)*(0.5*erfc(-((lowerj-rho(jj+1)*zi-mu_i(i_c,jj+1))-mew(jj+1))./(sigma_i(i_c,jj+1)*sqrt(2))));
    end
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);          % the lowest bin extends to -Inf
    P(:,znum)=1-P_part2(:,znum);  % the highest bin extends to +Inf
    pi_z_J(:,:,jj)=P;
end

%% Step 3: the age 1 distribution
% Whatever produced period 1, it is either a gaussian mixture or a normal (a normal being the
% one-component case), so put that distribution onto the age 1 grid using the same bin rule.
z1=z_grid_J(:,1);
if sigmaz(1)>0
    omega1=z1(2)-z1(1);
    up1=z1+omega1/2;
    lo1=z1-omega1/2;
    if j1ismixture==1
        p1=j1probs; m1=j1mu; s1=j1sigma;
    elseif isfield(tauchenoptions,'initialj1mewz') || isfield(tauchenoptions,'initialj1sigmaz')
        p1=1; m1=mewz(1); s1=sigmaz(1); % period 1 was set directly as a normal
    else
        % period 1 came from period 0: it is the age 1 mixture, shifted by mew(1)+rho(1)*z0, with
        % the period 0 variance added to every component
        p1=mixprobs_i(:,1);
        m1=mew(1)+rho(1)*z0+mu_i(:,1);
        s1=sqrt(rho(1)^2*sigmaz0^2+sigma_i(:,1).^2);
    end
    F1=zeros(znum,1); F0=zeros(znum,1);
    for i_c=1:length(p1)
        F1=F1+p1(i_c)*(0.5*erfc(-(up1-m1(i_c))./(s1(i_c)*sqrt(2))));
        F0=F0+p1(i_c)*(0.5*erfc(-(lo1-m1(i_c))./(s1(i_c)*sqrt(2))));
    end
    jequaloneDistz=F1-F0;
    jequaloneDistz(1)=F1(1);
    jequaloneDistz(znum)=1-F0(znum);
else
    % degenerate period 1: a point mass at mewz(1), which the grid above is centred on
    jequaloneDistz=zeros(znum,1);
    if rem(znum,2)==1
        jequaloneDistz((znum+1)/2)=1;
    else
        jequaloneDistz((znum/2):(znum/2+1))=[0.5;0.5];
    end
end
jequaloneDistz=jequaloneDistz/sum(jequaloneDistz);

%% The period 0 distribution, when period 0 was used
if sigmaz0>0
    z_grid_0=linspace(z0-Tauchen_q(1)*sigmaz0,z0+Tauchen_q(1)*sigmaz0,znum)';
    omega0=z_grid_0(2)-z_grid_0(1);
    G1=0.5*erfc(-((z_grid_0+omega0/2)-z0)./(sigmaz0*sqrt(2)));
    G0=0.5*erfc(-((z_grid_0-omega0/2)-z0)./(sigmaz0*sqrt(2)));
    jequalzeroDistz=G1-G0;
    jequalzeroDistz(1)=G1(1);
    jequalzeroDistz(znum)=1-G0(znum);
    jequalzeroDistz=jequalzeroDistz/sum(jequalzeroDistz);
else
    jequalzeroDistz=zeros(znum,1);
    if rem(znum,2)==1
        jequalzeroDistz((znum+1)/2)=1;
    else
        jequalzeroDistz((znum/2):(znum/2+1))=[0.5;0.5];
    end
end

%%
otheroutputs.sigma_z=sigmaz;
otheroutputs.mew_z=mewz;
otheroutputs.jequalzeroDistz=jequalzeroDistz;

if tauchenoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J);
    jequaloneDistz=gpuArray(jequaloneDistz);
end

end
