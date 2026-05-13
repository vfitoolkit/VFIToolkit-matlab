function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_e_raw(PolicyKron,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,z,semiz,e,j)

initialage=seedpoint(5); % j in (a,z,semiz,e,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

SimLifeCycleKron=nan(5,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
    SimLifeCycleKron(3,jj+initialage)=currstate(3); % z_c
    SimLifeCycleKron(4,jj+initialage)=currstate(4); % e

    d2ind=Policy_dsemiexo(currstate(1),currstate(2),currstate(3),currstate(4),jj+initialage);
    currstate(1)=PolicyKron(currstate(1),currstate(2),currstate(3),currstate(4),jj+initialage);

    [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,d2ind,jj+initialage)>rand(1,1));
    [~,currstate(3)]=max(cumsumpi_z_J(currstate(3),:,jj+initialage)>rand(1,1));
    [~,currstate(4)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
end
SimLifeCycleKron(5,initialage:(initialage+periods))=initialage:1:(initialage+periods);



end