function StationaryDistKron=StationaryDist_InfHorz_GI_raw(StationaryDistKron,Policy,l_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

% Options needed
%  simoptions.tolerance
%  simoptions.maxit
%  simoptions.multiiter

% First, get Gamma
Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
PolicyProbs=zeros(N_a,N_z,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.
Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1;
% aprimeProbs_lower=shiftdim((simoptions.ngridinterp+1-Policy(l_d+2,:,:)+1)/(simoptions.ngridinterp+1),1); % probability of lower grid point
% aprimeProbs_upper=shiftdim((Policy(l_d+2,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point
PolicyProbs(:,:,1)=shiftdim((simoptions.ngridinterp+1-Policy(l_d+2,:,:)+1)/(simoptions.ngridinterp+1),1); % probability of lower grid point
PolicyProbs(:,:,2)=1-PolicyProbs(:,:,1);

% R2025a, I can probably do this on gpu now??
Policy_aprime=gather(Policy_aprime);

StationaryDistKron=StationaryDist_InfHorz_Iteration_TwoProbs_raw(StationaryDistKron,Policy_aprime,PolicyProbs,N_a,N_z,pi_z,simoptions);

end
