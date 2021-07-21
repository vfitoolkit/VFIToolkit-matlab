function StationaryDist=StationaryDist_FHorz_Case1_Simulation(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z, simoptions)
%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

if nargin<6
    simoptions.nsims=10^4;
    simoptions.parallel=2;
    simoptions.verbose=0;
%    simoptions.ncores=1; not needed as using simoptions.parallel=2
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.nsims;','fieldexists=0;')
    if fieldexists==0
        simoptions.nsims=10^4;
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
PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,simoptions);

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);

StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,N_d,N_a,N_z,N_j,pi_z, simoptions);
    
StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
