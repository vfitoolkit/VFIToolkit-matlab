function TimeSeries=SimTimeSeriesValues_InfHorz(Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, d_grid, a_grid, z_grid,pi_z,simoptions)
% Simulate time series for the FnsToEvaluate for an infinite horizon model.

if ~exist(simoptions,'seedpoint')
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
else
    if ~isfield(simoptions,'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

simoptions.numbersims=1;
% To make my life easy, this actully just uses the SimPanelValues command and creates just one simulation.
InitialDist=zeros([N_a,N_z]);
InitialDist(simoptions.seedpoint)=1;
TimeSeries=SimPanelValues_Case1(InitialDist,Policy,FnsToEvaluate,[],Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions);

end
