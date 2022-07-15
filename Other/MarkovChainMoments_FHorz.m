function [mean,variance,autocorrelation,statdist]=MarkovChainMoments_FHorz(z_grid_J,pi_z_J,jequaloneDistz,simoptions,mcmomentsoptions)
% Calculates the mean, variance, autocorrellation, and stantionary
% distribution of a finite-horizon markov process.
% 
% Inputs:
%   - z_grid_J: the grids for each age; z_grid_J(:,jj) is the grid in period jj
%   - pi_z_J: the transition matrix for each age: pi_z_J(:,jj) is the transition matrix from period jj to period jj+1
%   - jequaloneDist: the distribution at age jj=1
% Optional inputs:
%   - mcmomentsoptions: note that if mcmomentsoptions contains z_grid_J, pi_z_J or jequaloneDist then these are used to overwrite the inputs
%   - mcmomentsoptions.n_z: if the markov process is multidimensional you will need to input n_z
%
% Outputs:
%   - mean: mean of first order markov chain for each age (i.e., age conditional mean)
%   - variance: variance of first order markov chain for each age  (i.e., age conditional variance)
%   - autocorrelation: autocorrelation of first order markov chain for each
%             age, j-th entry is autocorrelation between j and j-1 (hence j=1 is nan)
%   - statdist: stationary distribution of first order markov chain for
%             each age (note the age j=1 will just reproduce jequaloneDist)
%

if exist('mcmomentsoptions','var')==0
    mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    mcmomentsoptions.Tolerance=10^(-8);
    mcmomentsoptions.calcautocorrelation=1; % Have made it easy to skip calculating autocorrelation as this takes time
    mcmomentsoptions.calcautocorrelation_nsims=10^6; % Number of simulations to use for calculating the autocorrelation
    mcmomentsoptions.n_z=length(z_grid_J); % Is assumed to be a single valued z
else
    if isfield(mcmomentsoptions,'parallel')==0
        mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(mcmomentsoptions,'Tolerance')==0
        mcmomentsoptions.Tolerance=10^(-8);
    end
    if isfield(mcmomentsoptions,'calcautocorrelation')==0
        mcmomentsoptions.calcautocorrelation=1;
    end
    if isfield(mcmomentsoptions,'calcautocorrelation_nsims')==0
        mcmomentsoptions.calcautocorrelation_nsims=10^6;
    end
    if isfield(mcmomentsoptions,'n_z')==0
        mcmomentsoptions.n_z=length(z_grid_J); % Is assumed to be a single valued z
    end
end

%% Get pi_z_J z_grid_J and jequaloneDist
if ~exist('simoptions','var')
    simoptions=struct();
end

if isfield(simoptions,'jequaloneDist')
    if ~isempty(jequaloneDistz)
        warning('MarkovChainMoments_FHorz: Using simoptions.jequaloneDist to overwrite jequaloneDist')
    end
    jequaloneDistz=simoptions.jequaloneDist;
    jequaloneDistz=reshape(jequaloneDistz,[numel(jequaloneDistz),1]);
else
    if isempty(jequaloneDistz)
        error('MarkovChainMoments_FHorz: Need to input either simoptions.jequaloneDist or jequaloneDist')
    end
end
if isfield(simoptions,'pi_z_J')
    if ~isempty(pi_z_J)
        warning('MarkovChainMoments_FHorz: Using simoptions.pi_z_J to overwrite pi_z_J')
    end
    pi_z_J=simoptions.pi_z_J;
else
    if isempty(pi_z_J)
        error('MarkovChainMoments_FHorz: Need to input either simoptions.pi_z_J or pi_z_J')
    end
end
if isfield(simoptions,'z_grid_J')
    if ~isempty(z_grid_J)
        warning('MarkovChainMoments_FHorz: Using simoptions.z_grid_J to overwrite z_grid_J')
    end
    z_grid_J=simoptions.z_grid_J;
else
    if isempty(z_grid_J)
        error('MarkovChainMoments_FHorz: Need to input either simoptions.z_grid_J or z_grid_J')
    end
end

%% Get the dimensions
N_z=prod(mcmomentsoptions.n_z);
N_j=size(z_grid_J,2);

% Check the sizes of jequaloneDist and pi_z_J
if size(z_grid_J,1)~=sum(mcmomentsoptions.n_z)
    error('z_grid_J does not have right number of points for z')
end
if size(pi_z_J,1)~=N_z
    error('pi_z_J does not have right number of points for z')
elseif size(pi_z_J,2)~=N_z
    error('pi_z_J does not have right number of points for z')
elseif size(pi_z_J,3)~=N_j
    error('pi_z_J does not have right number of points for j (compared to z_grid_J)')
end
if length(jequaloneDistz)~=N_z
    error('z_grid_J and jequaloneDist disagree about the size of N_z')
end

if length(mcmomentsoptions.n_z)==1
    %% Compute the mean, variance, and (first-order auto-) correlation
    statdist=zeros(N_z,N_j);
    mean=zeros(1,N_j);
    secondmoment=zeros(1,N_j);
    variance=zeros(1,N_j);
    covar_withlag=nan(1,N_j); % e.g., the second entry is j=2 with j=1. First entry remains nan.
    autocorrelation=nan(1,N_j); % e.g., the second entry is j=2 with j=1. First entry remains nan.
    
    statdist(:,1)=jequaloneDistz;
    mean(1)=(z_grid_J(:,1)')*statdist(:,1);
    secondmoment(1)=(z_grid_J(:,1).^2)'*statdist(:,1);
    variance(1)=secondmoment(1)-mean(1).^2;
    for jj=2:N_j
        statdist(:,jj)=pi_z_J(:,:,jj-1)'*statdist(:,jj-1);
        
        mean(jj)=(z_grid_J(:,jj)')*statdist(:,jj);
        secondmoment(jj)=(z_grid_J(:,jj).^2)'*statdist(:,jj);
        
        variance(jj)=secondmoment(jj)-mean(jj).^2;
        
        covar_withlag(jj)=sum(statdist(:,jj-1).*sum(pi_z_J(:,:,jj-1).*((z_grid_J(:,jj-1)-mean(jj-1))*(z_grid_J(:,jj)-mean(jj))'),2));
        
        autocorrelation(jj)=covar_withlag(jj)/(sqrt(variance(jj))*sqrt(variance(jj-1)));
        
    end
    
else % z is multidimensional (note: I only calculate variance, autocorellation of each invidivually, not the actual covariance matrix and autocorrelation matrix of the multivariate)
    n_z=mcmomentsoptions.n_z;
    l_z=length(n_z);
    
    %% Compute the mean, variance, and (first-order auto-) correlation
    statdist=zeros(N_z,N_j);
    mean=zeros(l_z,N_j);
    secondmoment=zeros(l_z,N_j);
    variance=zeros(l_z,N_j);
    covar_withlag=nan(l_z,N_j); % e.g., the second entry is j=2 with j=1. First entry remains nan.
    autocorrelation=nan(l_z,N_j); % e.g., the second entry is j=2 with j=1. First entry remains nan.
    
    z_gridvals=CreateGridvals(n_z,z_grid_J(:,1),1);
    
    statdist(:,1)=jequaloneDistz;
    for ii=1:l_z
        mean(ii,1)=(z_gridvals(:,ii)')*statdist(:,1);
        secondmoment(ii,1)=(z_gridvals(:,ii).^2)'*statdist(:,1);
        variance(ii,1)=secondmoment(ii,1)-mean(ii,1).^2;
    end
    
    for jj=2:N_j
        z_gridvals_lag=z_gridvals;
        z_gridvals=CreateGridvals(n_z,z_grid_J(:,jj),1);

        statdist(:,jj)=pi_z_J(:,:,jj-1)'*statdist(:,jj-1);
        
        for ii=1:l_z
            mean(ii,jj)=(z_gridvals(:,ii)')*statdist(:,jj);
            secondmoment(ii,jj)=(z_gridvals(:,ii).^2)'*statdist(:,jj);
            
            variance(ii,jj)=secondmoment(ii,jj)-mean(ii,jj).^2;
            
            covar_withlag(ii,jj)=sum(statdist(:,jj-1).*sum(pi_z_J(:,:,jj-1).*((z_gridvals_lag(:,ii)-mean(ii,jj-1))*(z_gridvals(:,ii)-mean(ii,jj))'),2));
            
            autocorrelation(ii,jj)=covar_withlag(ii,jj)/(sqrt(variance(ii,jj))*sqrt(variance(ii,jj-1)));
        end
        
    end
end


%% Following is old code that used to simulate the markov to calculate the autocorrelation. 
% The final line checks it against the new code, and the new code is more
% accurate as well as immesurably faster.

% %% Now for the (first-order auto-) correlation
% % This takes vast majority of the time of MarkovChainMoments()
% % CAN THIS BE SPED UP WITH A DIRECT FORMULA FOR CORRELATION OF DISCRETE MARKOV
% N=mcmomentsoptions.calcautocorrelation_nsims; % Number of simulations to use for calculating the autocorrelation
% 
% if mcmomentsoptions.calcautocorrelation==1
%     
%     z_grid_J=gather(z_grid_J);
%         
%     cumjequaloneDist=cumsum(jequaloneDistz);
%     
%     cumsum_pi_z_J=cumsum(pi_z_J,2); % Sum in second dimension (next period state transition probabilities)
%     
%     % Simulate Markov chain with transition state pi_z
%     A=zeros(N,N_j); % A contains the time series of states
%     parfor ii=1:N
%         A_ii=zeros(1,N_j);
%         % jj=1
%         [~,A_ii(1)]=max(cumjequaloneDist>rand(1));
%         for jj=2:N_j
%             [~,zcurr]=max(cumsum_pi_z_J(A_ii(jj-1),:,jj-1)>rand(1));
%             A_ii(jj)=zcurr;
%         end
%         A(ii,:)=A_ii;
%     end
%     
%     correlation=nan(1,N_j); % Note the first entry will remain nan
%     for jj=2:N_j
%         temp=corrcoef(z_grid_J(A(:,jj),jj),z_grid_J(A(:,jj-1),jj));
%         correlation(jj)=temp(1,2);
%     end
% else
%     correlation=NaN;
% end
%  
% 
% [autocorrelation',correlation']


end