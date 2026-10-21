function TimeSeries=SimTimeSeriesValues_InfHorz(Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, d_grid, a_grid, z_grid,pi_z,simoptions)
% Simulate time series for the FnsToEvaluate for an infinite horizon model.

N_a=prod(n_a);
N_z=prod(n_z);

if ~exist('simoptions','var')
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.burnin=50;
else
    if ~isfield(simoptions,'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if ~isfield(simoptions,'burnin')
        simoptions.burnin=50;
    end
end
simoptions.numbersims=1;

if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

if isfield(simoptions,'n_semiz')
    N_semiz=prod(simoptions.n_semiz);
else
    N_semiz=0;
end

% To make my life easy, this actually just uses the SimPanelValues command and creates just one simulation.
if N_semiz>0
    % The seedpoint is for (a,z), the initial semiz is spread evenly over the semiz points, and the initial e is drawn from its iid distribution pi_e
    if N_z==0
        InitialDist=zeros([N_a,N_semiz]);
        InitialDist(simoptions.seedpoint(1),:)=1/N_semiz;
    else
        InitialDist=zeros([N_a,N_semiz,N_z]);
        InitialDist(simoptions.seedpoint(1),:,simoptions.seedpoint(2))=1/N_semiz;
    end
    if N_e>0
        InitialDist=reshape(InitialDist,[N_a,N_semiz,max(N_z,1)]).*shiftdim(gather(simoptions.pi_e(:)),-3); % [N_a,N_semiz,N_z,N_e]
    end
elseif N_z==0 && N_e==0
    InitialDist=zeros([N_a,1]);
    InitialDist(simoptions.seedpoint(1))=1;
elseif N_z>0 && N_e==0
    InitialDist=zeros([N_a,N_z]);
    InitialDist(simoptions.seedpoint(1),simoptions.seedpoint(2))=1;
elseif N_z==0 && N_e>0
    % The seedpoint is just for a, and the initial e is drawn from its iid distribution pi_e
    InitialDist=zeros([N_a,N_e]);
    InitialDist(simoptions.seedpoint(1),:)=gather(simoptions.pi_e(:))';
else % N_z>0 && N_e>0
    % The seedpoint is for (a,z), and the initial e is drawn from its iid distribution pi_e
    InitialDist=zeros([N_a,N_z,N_e]);
    InitialDist(simoptions.seedpoint(1),simoptions.seedpoint(2),:)=shiftdim(gather(simoptions.pi_e(:)),-2);
end

TimeSeries=SimPanelValues_InfHorz(InitialDist,Policy,FnsToEvaluate,[],Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions);

end
