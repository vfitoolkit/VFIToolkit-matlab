function StationaryDist=StationaryDist_InfHorz_GI_e_raw(StationaryDist,Policy,n_d,n_a,N_a,N_z,N_e,pi_z,pi_e,simoptions)
% Will treat the agents as being on a continuum of mass 1.
% Uses the improvement of: Tan (2020) - A fast and low computational memory algorithm for non-stochastic simulations in heterogeneous agent models

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

Policy=reshape(Policy, [size(Policy,1),N_a,N_z*N_e]);

% First, get Gamma
Policy_aprime=zeros(N_a,N_z*N_e,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
PolicyProbs=zeros(N_a,N_z*N_e,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

if l_a==1
    Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
elseif l_a==2
    Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
elseif l_a==3
    Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
elseif l_a==4
    Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
else
    error('StationaryDist_InfHorz_GI_raw cannot handle length(n_a)>4, contact me if you need this')
end
Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

PolicyProbs(:,:,2)=shiftdim((Policy(end,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point
PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

StationaryDist=StationaryDist_InfHorz_Iteration_nProbs_e_raw(StationaryDist,Policy_aprime,PolicyProbs,2,N_a,N_z,N_e,pi_z,pi_e,simoptions);

end
