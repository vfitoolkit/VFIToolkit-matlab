function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_AgeDepGrids_Dynasty_raw(Phi_of_Policy,Case2_Type,daz_gridstructure, N_j, seedpoint,simperiods)
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

SimLifeCycleKron=nan(2,seedpoint(3)+simperiods-1); % This is changed for dynasty

currstate=seedpoint;

% seedpoint is (a,z,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=simperiods; % This is changed for dynasty

for tt=1:periods
    jj=seedpoint(3)+tt-1;
    agej=rem(jj,N_j); % This is changed for dynasty
    if agej==0
        agej=N_j;
    end    
    jstr=daz_gridstructure.jstr{agej}; % This is changed for dynasty
    
    SimLifeCycleKron(1,jj)=currstate(1); %a_c
    SimLifeCycleKron(2,jj)=currstate(2); %z_c
    
    cumsumpi_z=daz_gridstructure.cumsumpi_z.(jstr(:));
    [~,zprimestate]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
    Phi_of_Policy_jj=Phi_of_Policy.(jstr(:));
    if Case2_Type==1 % phi(d,a,z,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
        % Create P matrix
    elseif Case2_Type==11 % phi(d,a,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimLifeCycleIndexes_FHorz_Case2_raw)')
    elseif Case2_Type==12 % phi(d,a,z)
        currstate(1)=Phi_of_Policy_jj(currstate(1),currstate(2));%,jj+seedpoint(3)-1);
    elseif Case2_Type==2  % phi(d,z',z)
        currstate(1)=Phi_of_Policy_jj(currstate(1),zprimestate,currstate(2));%,jj+seedpoint(3)-1);
    end
    currstate(2)=zprimestate;
end
