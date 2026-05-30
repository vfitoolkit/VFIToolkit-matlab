function [e_grid,pi_e]=discretizeIID_Tauchen(mew,sigma,enum,Tauchen_q, tauchenoptions)
% Create states vector, e_grid, and probability vector, pi_e, for the discrete approximation
%    of iid process e~N(mew,sigma^2), by Tauchen method
%
% Inputs
%   mew            - mean
%   sigma          - standard deviation
%   enum           - number of states in discretization of e (must be an odd number)
%   Tauchen_q      - (Hyperparameter) Defines max/min grid points as mew+-Tauchen_q*sigma (I suggest 2 or 3)
% Optional Inputs (tauchenoptions)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   e_grid         - column vector containing the enum states of the discrete approximation of e
%   pi_e           - column vector of probabilities of the discrete approximation of e;
%                    pi_e(i) is the probability of state i (sums to 1)
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
    e_grid=mew*ones(enum,1) + linspace(-Tauchen_q*sigma,Tauchen_q*sigma,enum)';
    omega=e_grid(2)-e_grid(1); %Note that all the points are equidistant by construction.

    P_part1=normcdf(e_grid+omega/2,mew,sigma);
    P_part2=normcdf(e_grid-omega/2,mew,sigma);

    pi_e=P_part1-P_part2;
    pi_e(1)=P_part1(1);
    pi_e(enum)=1-P_part2(enum);

elseif tauchenoptions.parallel==2 %Parallelize on GPU
    e_grid=gpuArray(mew*ones(enum,1) + linspace(-Tauchen_q*sigma,Tauchen_q*sigma,enum)');
    omega=e_grid(2)-e_grid(1); %Note that all the points are equidistant by construction.

    %Note: normcdf is not yet a supported function for use on the gpu in Matlab
    %However erf is supported, and we can easily construct our own normcdf
    %from erf (see http://en.wikipedia.org/wiki/Normal_distribution for the
    %formula for normcdf as function of erf)

    erfinput=arrayfun(@(ei,omega,mew,sigma) ((ei+omega/2)-mew)/sqrt(2*sigma^2), e_grid,omega,mew,sigma);
    P_part1=0.5*(1+erf(erfinput));

    erfinput=arrayfun(@(ei,omega,mew,sigma) ((ei-omega/2)-mew)/sqrt(2*sigma^2), e_grid,omega,mew,sigma);
    P_part2=0.5*(1+erf(erfinput));

    pi_e=P_part1-P_part2;
    pi_e(1)=P_part1(1);
    pi_e(enum)=1-P_part2(enum);

end

end
