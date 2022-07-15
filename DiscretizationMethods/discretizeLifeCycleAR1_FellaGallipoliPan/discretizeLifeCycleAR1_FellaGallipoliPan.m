function [z_grid_J, P_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_FellaGallipoliPan(rho,sigma,znum,J,fellagallipolipanoptions)
% Please cite: Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Fella-Gallipoli-Pan discretization method for a 'life-cycle non-stationary AR(1) process'. 
% This is an extension of the Rouwenhorst method to 'age-dependent parameters'
% 
%  Exteneded-Rouwenhurst method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1)) 
%
% Inputs:  
%   rho 	     - Jx1 vector of serial correlation coefficients
%   sigma        - Jx1 vector of standard deviations of innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (fellagallipolipanoptions)
%   parallel:    - set equal to 2 to use GPU, 0 to use CPU
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   P            - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%                  P(:,:,j) is transition matrix from age j to j+1 (Modified from FGP where it is j-1 to j)
%   jequaloneDistz - znum-by-1 vector, the distribution of z in period 1
%   otheroutputs - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%
% This code is by Fella, Gallipoli & Pan.
% Lightly modified by Robert Kirkby.
% !========================================================================%
% Original paper:
% Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Two changes from FGP2019:
%    i) Allow z0 to be a normal distribution, rather than forcing z0=0;
%    using fellagallipolipanoptions.initialj0sigma_z
%    ii) Here P_J(:,:,j) is transition from j to j+1; in FGP2019 it was from j-1 to j.
%
% FGP use MIT license, which must be included with the code, you can find it at the bottom of this 
% script. (VFI Toolkit is GPL3 license, hence having to reproduce.)

sigma_z = zeros(1,J);
% z_grid = zeros(znum,J); 
P_J = zeros(znum,znum,J);

%% Set options
if ~exist('fellagallipolipanoptions','var')
    fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    fellagallipolipanoptions.nSigmas=min(sqrt(znum-1),4);
else
    if ~isfield(fellagallipolipanoptions,'parallel')
        fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(fellagallipolipanoptions,'nSigmas')
        fellagallipolipanoptions.nSigmas=min(sqrt(znum-1),4);
    end
end

%% Check inputs
if znum < 2
    error('The state space has to have dimension znum>1. Exiting.')
end

if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

%% Step 1: construct the state space z_grid_J for each period j.
% Evenly-spaced znum-state space over [-kirkbyoptions.nSigmas*sigma_z(j),kirkbyoptions.nSigmas*sigma_z(j)].

% 1.a Compute unconditional variances of z(j)
if isfield(fellagallipolipanoptions,'initialj0sigma_z')
    sigma_z(1) = sqrt(rho(1)^2*fellagallipolipanoptions.initialj0sigma_z^2+sigma(1)^2);
else
    sigma_z(1) = sigma(1);
end

for jj = 2:J
    sigma_z(jj) = sqrt(rho(jj)^2*sigma_z(jj-1)^2+sigma(jj)^2);
end

% 1.b Construct state space
h = 2*fellagallipolipanoptions.nSigmas*sigma_z/(znum-1); % grid step (2* as is nSigmas either side of zero)
z_grid_J = repmat(h,znum,1);
z_grid_J(1,:)=-fellagallipolipanoptions.nSigmas*sigma_z;
z_grid_J = cumsum(z_grid_J,1); 

%% Step 2: Compute the transition matrices trans(:,:,t) from period (t-1) to period t
% The transition matrix for period t is defined by parameter p(t).
% p(t) = 0.5*(1+rho*sigma(t-1)/sigma(t))

% Note: P(:,:,1) is the transition matrix from z(0)=0 to any gridpoint of z_grid(1) in period 1.
% Any of its rows is the (unconditional) distribution in period 1.

% Note: rhmat() is the 'Rouwenhorst matrix' subfunction

p = 1/2; % First period: p(1) = 0.5 as y(1) is white noise.  
P_J(:,:,1) = rhmat(p,znum);

for jj = 2:J
    % Compute p for t>1
    p = (sigma_z(jj)+rho(jj)*sigma_z(jj-1))/(2*sigma_z(jj));
    P_J(:,:,jj) = rhmat(p,znum);
    % Note that here P_J(:,:,jj) is the transition from jj-1 into jj
end

%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if fellagallipolipanoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    P_J=gpuArray(P_J);
end

%%
jequaloneDistz=P_J(1,:,1)';

%% P(:,:,j) is transition from age j to j+1 (Modified from FGP where it is j-1 to j)
% Change P_J so that P_J(:,:,jj) is the transition matrix from period jj to period jj+1
P_J(:,:,1:end-1)=P_J(:,:,2:end);
% For jj=J, P_J(:,:,J) is kind of meaningless (there is no period jj+1 to transition to). I just fill it in as a uniform distribution
P_J(:,:,J)=ones(znum,znum)/znum;


%% Subfunction rhmat()
function [Pmat] = rhmat(p,N)
    % Computes Rouwenhorst matrix as a function of p and N
    Pmat = zeros(N,N);
    % Step 2(a): get the transition matrix P1 for the N=2 case
    if N == 2
        Pmat = [p, 1-p; 1-p, p];
    else
        P1 = [p, 1-p; 1-p, p];
        % Step 2(b): if the number of states N>2, apply the Rouwenhorst
        % recursion to obtain the transition matrix trans
        for ii = 2:N-1
            P2 = p *     [P1,zeros(size(P1,1),1); zeros(1,size(P1,2)),0 ] + ...
                (1-p) * [zeros(size(P1,1),1),P1; 0,zeros(1,size(P1,2)) ] + ...
                (1-p) * [zeros(1,size(P1,2)),0 ; P1,zeros(size(P1,1),1)] + ...
                p *     [0,zeros(1,size(P1,2)) ; zeros(size(P1,1),1),P1];
            
            P2(2:ii,:) = 0.5*P2(2:ii,:);
            
            if ii==N-1
                Pmat = P2;
            else
                P1 = P2;
            end
        end % of for
    end % if N == 2
end % of rhmat function

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.sigma_z=sigma_z; % Standard deviation of z (for each period)

end


%%
% The MIT License (MIT)
% 
% Copyright (c) 2019 Giulio Fella, Giovanni Gallipoli and Jutong Pan
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
%     If you use this library or parts of it in your work, we request that you cite the package. A suggested citation is
% 
%     "nmarkov-matlab: Markov-chain approximations for non-stationary AR(1) processes (Matlab version)" https://github.com/gfell/nsmarkov-matlab based on the paper "Markov-Chain Approximations for Life-Cycle Models"
%     by Giulio Fella, Giovanni Gallipoli and Jutong Pan, Review of Economic Dynamics 34, 2019 (https://doi.org/10.1016/j.red.2019.03.013).
% 
%     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
