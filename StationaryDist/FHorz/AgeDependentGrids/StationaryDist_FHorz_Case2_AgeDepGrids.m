function StationaryDist=StationaryDist_FHorz_Case2_AgeDepGrids(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions)
% Do not call this command directly, it is a subcommand of StationaryDist_Case2_FHorz()

%%
daz_gridstructure=AgeDependentGrids_Create_daz_gridstructure(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Parameters, simoptions);
% Creates daz_gridstructure which contains both the grids themselves and a
% bunch of info about the grids in an easy to access way.
% e.g., the d_grid for age j=10: daz_gridstructure.d_grid.j010
% e.g., the value of N_a for age j=5: daz_gridstructure.N_a.j005
% e.g., the zprime_grid for age j=20: daz_gridstructure.zprime_grid.j020

%%
if simoptions.verbose==1
    simoptions
end

jequaloneDistKron=reshape(jequaloneDist,[daz_gridstructure.N_a.j001*daz_gridstructure.N_z.j001,1]);

% Transform Policy into kroneckered form
PolicyKron=struct();
for jj=1:N_j
    % Make a three digit number out of jj
    if jj<10
        jstr=['j00',num2str(jj)];
    elseif jj>=10 && jj<100
        jstr=['j0',num2str(jj)];
    else
        jstr=['j',num2str(jj)];
    end
    n_d_j=daz_gridstructure.n_d.(jstr(:));
    n_a_j=daz_gridstructure.n_a.(jstr(:));
    n_z_j=daz_gridstructure.n_z.(jstr(:));

    if simoptions.parallel==2
        PolicyKron.(jstr)=KronPolicyIndexes_Case2(Policy.(jstr), n_d_j', n_a_j', n_z_j');%,simoptions); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    else % Often Policy will be on gpu but want to iterate on StationaryDist using sparse matrices on cpu. Hence have taken this approach which allows 'Kron' to be done on gpu, where Policy is, and then moved to cpu. 
        PolicyKron.(jstr)=gather(KronPolicyIndexes_Case2(Policy.(jstr), n_d_j', n_a_j', n_z_j'));%,simoptions)); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    end
end

if simoptions.dynasty==1
    if simoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    % With dynasty, jequaloneDistKron is really just an initial guess, and should have no influence on the eventual outcome.
    if simoptions.iterate==0
        fprintf('Simulating the stationary agents distribution has not yet been implemented for Case2 of FHorz, \n please email me if you have a need for it, otherwise use simoptions.iterate=1 to iterate the stationary distribution \n')
        % StationaryDistKron=StationaryDist_FHorz_Case2_AgeDependentGrids_Dynasty_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    elseif simoptions.iterate==1
        StationaryDistKron=StationaryDist_FHorz_Case2_AgeDepGrids_Dynasty_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
    end
else
    if simoptions.iterate==0
        fprintf('Simulating the stationary agents distribution has not yet been implemented for Case2 of FHorz, \n please email me if you have a need for it, otherwise use simoptions.iterate=1 to iterate the stationary distribution \n')
%         StationaryDistKron=StationaryDist_FHorz_Case2_AgeDependentGrids_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Params,PhiaprimeParamNames,simoptions);
    elseif simoptions.iterate==1
        StationaryDistKron=StationaryDist_FHorz_Case2_AgeDepGrids_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
    end
end


% StationaryDist out of kroneckered form
StationaryDist=struct();
for jj=1:N_j
    % Make a three digit number out of jj
    jstr=daz_gridstructure.jstr{jj};
    n_d_j=daz_gridstructure.n_d.(jstr);
    n_a_j=daz_gridstructure.n_a.(jstr);
    n_z_j=daz_gridstructure.n_z.(jstr);

    StationaryDist.(jstr)=reshape(StationaryDistKron.(jstr),[n_a_j,n_z_j]); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    StationaryDist.AgeWeights=StationaryDistKron.AgeWeights;
end

end
