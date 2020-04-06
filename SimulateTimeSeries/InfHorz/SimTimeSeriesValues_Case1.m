function SimTimeSeries=SimTimeSeriesValues_Case1(Policy,n_d,n_a,n_z,pi_z, ContinuousVarIndicator_d, ContinuousVarIndicator_a, simoptions)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1; % Which form is Policy inputted as?
    simoptions.burnin=1000;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10000;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.pi_z=1; % 1: z is discrete and pi_z is transition matrix, 
                       % 2: pi_z is a function that takes last period state vector as input, and returns current state vector as output 
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'burnin')
        simoptions.burnin=1000;
    end
    if ~isfield(simoptions, 'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=10^4;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
end

%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins from point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)


if N_d>0
    l_d=length(n_d);
else 
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);

%Policy is [l_d+l_a,n_a,n_s,n_z]

if simoptions.polindorval==1
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,2); % Assumes Policy was calculated on GPU
end

Policy=gather(Policy);
pi_z=gather(pi_z);


seedtemp=sub2ind_homemade([n_a,n_z],simoptions.seedpoint);
seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);
seedpoint_val=nan(size(seedpoint));
for ii=1:l_a
    if ii==1
        seedpoint_val(ii)=a_grid(seedpoint(ii));
    else
        seedpoint_val(ii)=a_grid(seedpoint(ii)+sum(n_a(1:ii-1)));
    end
end
for ii=1:l_z
    if ii==1
        seedpoint_val(ii+l_a)=z_grid(seedpoint(ii+l_a));
    else
        seedpoint_val(ii+l_a)=z_grid(seedpoint(ii+l_a)+sum(n_z(1:ii-1)));
    end
end

% currstate=seedpoint;
SimTimeSeries=nan(l_d+l_a+l_z,simoptions.burnin+simoptions.simperiods);

% First, just simulate all the z values (I do this seperately to allow for
% pi_z to be either an actual transition matrix, or a function.
if simoptions.pi_z==1
    zTimeSeries_ind=nan(1,simoptions.burnin+simoptions.simperiods);
    cumsum_pi_z=cumsum(pi_z,2);
    z_grid_shift=[0,cumsum(n_z(1:end-1))];
    current_z_ind=seedpoint(2);
    for t=1:simoptions.burnin+simoptions.simperiods
        [~,current_z_ind]=max(cumsum_pi_z(current_z_ind,:)>rand(1,1));
        zTimeSeries_ind(t)=current_z_ind;
        z_vec=ind2sub_homemade([n_z],current_z_ind);
        SimTimeSeries(l_d+l_a+1:end,t)=z_grid(z_vec+z_grid_shift);
    end
elseif simoptions.pi_z==2
    % pi_z=@(z) exp(rho*log(z)+sqrt(sigmasq_epsilon)*randn(1,1))
    current_z_val=seedpoint_val(l_a+1:end);
    for t=1:simoptions.burnin+simoptions.simperiods
        current_z_val=pi_z(current_z_val);
        SimTimeSeries(l_d+l_a+1:end,t)=current_z_val;
    end
end
    
if n_d(1)==0
    current_a_val=seedpoint_val(1:l_a);
%%    %AM UP TO HERE!!!!!!!
% Need to use a combination of ContinuousVarIndicator_d,
% ContinuousVarIndicator_a, 'squeeze' and 'interpn' to do this.
    for ii=1:l_a
        a_val_ii = interpn(X1,X2,Xn,Policy(ii,),Xq1,Xq2,Xqn);
    end
else %ie., that n_d(1)~=0
    current_a_val=seedpoint_val(1:l_a);
end

if simoptions.parallel==2
    SimTimeSeries=gpuArray(SimTimeSeries);
end


end



