function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_noz_raw(PolicyKron,Policy_dsemiexo,N_j,cumsumpi_semiz_J,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,semiz,j)

initialage=seedpoint(3); % j in (a,semiz,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

SimLifeCycleKron=nan(3,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c

    d2ind=Policy_dsemiexo(currstate(1),currstate(2),jj+initialage);
    currstate(1)=PolicyKron(currstate(1),currstate(2),jj+initialage); % (d2,aprime)
    
    [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,d2ind,jj+initialage)>rand(1,1));
end
SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);

end