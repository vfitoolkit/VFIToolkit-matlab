function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_raw(Phi_of_Policy,Case2_Type,N_d,N_a,N_z,N_j,cumsumpi_z,seedpoint,simperiods,fieldexists_ExogShockFn)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z) for every period j. This is for period 1 to
% J. Since most simulations will not start at period 1, the first entries
% are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

SimLifeCycleKron=nan(2,N_j);

currstate=seedpoint;

% seedpoint is (a,z,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j+1-seedpoint(3));

if fieldexists_ExogShockFn==0
    for jj=1:periods
        SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); %a_c
        SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); %z_c
        
        [~,zprimestate]=max(cumsumpi_z(currstate(2),:)>rand(1,1)); %max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
        if Case2_Type==1 % phi(d,a,z,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
            % Create P matrix
        elseif Case2_Type==11 % phi(d,a,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
        elseif Case2_Type==12 % phi(d,a,z)
            currstate(1)=Phi_of_Policy(currstate(1),currstate(2),jj+seedpoint(3)-1);        
        elseif Case2_Type==2  % phi(d,z',z)
            currstate(1)=Phi_of_Policy(currstate(1),zprimestate,currstate(2),jj+seedpoint(3)-1);
        end
        currstate(2)=zprimestate;
    end
else
    for jj=1:periods
        SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); %a_c
        SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); %z_c

        [~,zprimestate]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
        if Case2_Type==1 % phi(d,a,z,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
            % Create P matrix
        elseif Case2_Type==11 % phi(d,a,z')
            disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
        elseif Case2_Type==12 % phi(d,a,z)
            currstate(1)=Phi_of_Policy(currstate(1),currstate(2),jj+seedpoint(3)-1);        
        elseif Case2_Type==2  % phi(d,z',z)
            currstate(1)=Phi_of_Policy(currstate(1),zprimestate,currstate(2),jj+seedpoint(3)-1);
        end
        currstate(2)=zprimestate;
    end
end
