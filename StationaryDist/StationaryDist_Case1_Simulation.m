function StationaryDist=StationaryDist_Case1_Simulation(Policy,n_d,n_a,n_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

if nargin<6
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
%    simoptions.ncores=1; not needed as using simoptions.parallel=2
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=10^4;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=10^3;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    if simoptions.parallel>0
        eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
        if fieldexists==0
            simoptions.ncores=NCores;
        end
    end
end

%%
PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z); %,simoptions);

StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
    
StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

end
