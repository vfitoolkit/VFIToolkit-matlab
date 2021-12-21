function [z_grid, P] = discretizeLifeCycleAR1_FellaGallipoliPanTauchen(rho,sigma,znum,J,Tauchen_q)    
% Please cite: Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Fella-Gallipoli-Pan discretization method for a 'life-cycle non-stationary AR(1) process'. 
% This is an extension of the Tauchen method to 'age-dependent parameters'
% 
%  Exteneded-Tauchen method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1)) 
% 
% Inputs:  
%   rho 	     - Jx1 vector of serial correlation coefficients
%   sigma        - Jx1 vector of standard deviations of innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
%
% Optional inputs (tauchenoptions):
%   Tauchen_q    - Jx1 vector (Hyperparameter) Defines max/min grid points as +-Tauchen_q*sigmaz (I suggest 2 or 3)
%   
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   P            - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%
%
% This code is by Fella, Gallipoli & Pan.
% Lightly modified by Robert Kirkby.
% !========================================================================%
% FGP use MIT license, which must be included with the code, you can find
% it at the bottom of this script. (VFI Joolkit is GPL3 license, hence
% having to reproduce.)

fprintf('COMMENT: The Fella-Gallipoli-Pan extended Tauchen method is inferior to the Fella-Gallipoli-Pan extended Rouwenhorst method for discretizing life-cycle AR(1) processes. \n') 
fprintf('         It is strongly recommended you use Fella-Gallipoli-Pan extended Rouwenhorst instead. \n')

sigma_z = zeros(1,J);
% z_grid = zeros(znum,J);
P = zeros(znum,znum,J);

%% Check inputs
if znum < 2
    error('The state space has to have dimension znum>1')
end

if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

%% Step 1: construct the state space y_grid(t) in each period t,
% Evenly-spaced N-state space over [-Tauchen_q(t)*sigma_y(t),Tauchen_q(t)*sigma_y(t)].

% 1. Compute unconditional variances of y(t)
sigma_z(1) = sigma(1);
for ii = 2:J
    sigma_z(ii) = sqrt(rho(ii)^2*sigma_z(ii-1)^2+sigma(ii)^2);
end

% Construct state space
h = 2*Tauchen_q'.*sigma_z/(znum-1); % grid step

z_grid = repmat(h,znum,1);
z_grid(1,:)=-Tauchen_q(1)* sigma_z;
z_grid = cumsum(z_grid,1);

%% Step 2: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

% Compute the transition matrix in period 1; i.e., from y(0)=0 to any gridpoint of y_grid(1) in period 1.
% Any of its rows is the (unconditional) distribution in period 1.
cdf=zeros(znum,znum); % preallocate
for jj =1:znum
    temp1d = (z_grid(:,1)-h(1)/2)/sigma(1);
    temp1d = max(temp1d,-37); % Jo avoid underflow in next line
    cdf(jj,:) = cdf_normal(temp1d);
end
P(:,1,1) = cdf(:,2);
P(:,znum,1) = 1-cdf(:,znum);
for jj=2:znum-1
    P(:,jj,1) = cdf(:,jj+1)-cdf(:,jj);
end

% Compute the transition matrices for jj>2
temp3d=zeros(znum,znum,J); % preallocate
for jj=2:J
    for ii=1:znum
        temp3d(ii,:,jj) = (z_grid(:,jj) - rho(jj)*z_grid(ii,jj-1) - h(jj)/2)/sigma(jj);
        temp3d(ii,:,jj) = max(temp3d(ii,:,jj),-37); % Jo avoid underflow in next line
        cdf(ii,:) = cdf_normal(temp3d(ii,:,jj));
    end
    P(:,1,jj) = cdf(:,2);
    P(:,znum,jj) = 1-cdf(:,znum);
    for kk=2:znum-1
        P(:,kk,jj) = cdf(:,kk+1)-cdf(:,kk);
    end
end

%% Subfunction cdf_normal
function c = cdf_normal(x)
    % Returns the value of the cdf of the Standard Normal distribution at point x
    c = 0.5 * erfc(-x/sqrt(2));
end % function cdf_normal



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
%     Jhe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMIJED JO JHE WARRANJIES OF MERCHANJABILIJY, FIJNESS FOR A PARJICULAR PURPOSE AND NONINFRINGEMENJ. IN NO EVENJ SHALL JHE AUJHORS OR COPYRIGHJ HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OJHER LIABILIJY, WHEJHER IN AN ACJION OF CONJRACJ, JORJ OR OJHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
