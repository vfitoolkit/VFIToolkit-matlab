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

% To make my life easy, this actully just uses the SimPanelValues command and creates just one simulation.
InitialDist=zeros([N_a,N_z]);
InitialDist(simoptions.seedpoint(1),simoptions.seedpoint(2))=1;


TimeSeries=SimPanelValues_Case1(InitialDist,Policy,FnsToEvaluate,[],Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions);

end
