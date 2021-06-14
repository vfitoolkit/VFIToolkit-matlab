function SimPanelValues=SimPanelValues_FHorz_Case2_AgeDepGrids_Dynasty(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_gridfn,a_gridfn,z_gridfn,AgeDependentGridParamNames, Phi_aprimeFn,Case2_Type,PhiaprimeParamNames, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

daz_gridstructure=AgeDependentGrids_Create_daz_gridstructure(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Parameters, simoptions);
% Creates daz_gridstructure which contains both the grids themselves and a
% bunch of info about the grids in an easy to access way.
% e.g., the d_grid for age j=10: daz_gridstructure.d_grid.j010
% e.g., the value of N_a for age j=5: daz_gridstructure.N_a.j005
% e.g., the zprime_grid for age j=20: daz_gridstructure.zprime_grid.j020
% e.g., the jstr for age j=15: daz_gridstructure.jstr(15)='j015'

% NOTE: ESSENTIALLY ALL THE RUN TIME IS IN THIS COMMAND. WOULD BE GOOD TO OPTIMIZE/IMPROVE.
% Transform Policy into kroneckered form
PolicyIndexesKron=struct();
for jj=1:N_j
    jstr=daz_gridstructure.jstr{jj};
    n_d_j=daz_gridstructure.n_d.(jstr(:));
    n_a_j=daz_gridstructure.n_a.(jstr(:));
    n_z_j=daz_gridstructure.n_z.(jstr(:));

    if simoptions.parallel==2
        PolicyIndexesKron.(jstr)=KronPolicyIndexes_Case2(Policy.(jstr), n_d_j', n_a_j', n_z_j');%,simoptions); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    else % Often Policy will be on gpu but want to iterate on StationaryDist using sparse matrices on cpu. Hence have taken this approach which allows 'Kron' to be done on gpu, where Policy is, and then moved to cpu. 
        PolicyIndexesKron.(jstr)=gather(KronPolicyIndexes_Case2(Policy.(jstr), n_d_j', n_a_j', n_z_j'));%,simoptions)); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    end
end

SimPanelIndexes=SimPanelIndexes_FHorz_Case2_AgeDepGrids_Dynasty(InitialDist,PolicyIndexesKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames, simoptions);

% PolicyIndexesKron=gather(PolicyIndexesKron); (done below as needs to be in loop over N_j)

SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

%% Precompute the gridvals vectors.
% daz_gridvals=struct();
dPolicy_gridvals=struct();
l_d=length(n_d_j); % Note that l_d cannot vary with j
l_a=length(n_a_j); % Note that l_a cannot vary with j
l_z=length(n_z_j); % Note that l_z cannot vary with j
for jj=1:N_j
    jstr=daz_gridstructure.jstr{jj};
    n_d_j=daz_gridstructure.n_d.(jstr(:));
%     n_aprime_j=daz_gridstructure.n_aprime.(jstr(:));
    n_a_j=daz_gridstructure.n_a.(jstr(:));
    n_z_j=daz_gridstructure.n_z.(jstr(:));
    d_grid_j=gather(daz_gridstructure.d_grid.(jstr(:)));
    
%     aprime_grid_j=gather(daz_gridstructure.aprime_grid.(jstr(:)));

    [dPolicy_gridvals_j, ~]=CreateGridvals_PolicyKron(PolicyIndexesKron.(jstr),n_d_j,[],n_a_j,n_z_j,d_grid_j,[],2,1);
    dPolicy_gridvals.(jstr(:))=dPolicy_gridvals_j;
    
    daz_gridstructure.d_grid.(jstr(:))=gather(daz_gridstructure.d_grid.(jstr(:)));
%     n_a_j=daz_gridstructure.n_a.(jstr(:));
%     n_z_j=daz_gridstructure.n_z.(jstr(:));
%     N_a_j=daz_gridstructure.N_a.(jstr(:));
%     N_z_j=daz_gridstructure.N_z.(jstr(:));
%     a_grid_j=gather(daz_gridstructure.a_grid.(jstr(:)));
%     z_grid_j=gather(daz_gridstructure.z_grid.(jstr(:)));

%     daz_gridstructure.d_grid.(jstr(:))=gather(daz_gridstructure.d_grid.(jstr(:)));
%     PolicyIndexesKron.(jstr)=gather(PolicyIndexesKron.(jstr));
end

%%
SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods);
%% For sure the following could be made faster by parallelizing some stuff.
% WITH AGE DEPENDENT grids it would be much faster to first sort all the
% observations of SimPanel_ii by age j. Then go through age by age giving
% the values. Then unsort again (reverse the sorting). BUT RIGHT NOW I AM
% TOO LAZY TO DO THIS!!!! SimPanelIndexes_FHorz_Case2_AgeDepGrids CONTAINS
% AN EXAMPLE OF HOW TO DO IT. (could then also parfor across the different
% ages to make things even faster!)

for ii=1:simoptions.numbersims
    SimPanel_ii=SimPanelIndexes(:,:,ii);

    for t=1:simoptions.simperiods
        j_ind=SimPanel_ii(end,t);

        if ~isnan(j_ind) % isnan(j_ind) means that the agent reached the end of their finite lifetime and has 'died'
            jstr=daz_gridstructure.jstr{j_ind};
%             n_d_j=daz_gridstructure.n_d.(jstr(:));
            N_a_j=daz_gridstructure.N_a.(jstr(:));
            N_z_j=daz_gridstructure.N_z.(jstr(:));
            n_a_j=daz_gridstructure.n_a.(jstr(:));
            n_z_j=daz_gridstructure.n_z.(jstr(:));
            
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a_j,a_sub);
%             a_val=daz_gridvals(j_ind).a_gridvals_j(a_ind,:);
            a_gridvals_j=daz_gridstructure.a_gridvals.(jstr(:)); %Old: daz_gridvals(j_ind).a_gridvals_j(a_ind,:);
            a_val=a_gridvals_j(a_ind,:);
            
            z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
            z_ind=sub2ind_homemade(n_z_j,z_sub);
%             z_val=daz_gridvals(j_ind).z_gridvals_j(z_ind,:);
            z_gridvals_j=daz_gridstructure.z_gridvals.(jstr(:)); %Old: daz_gridvals(j_ind).z_gridvals_j(z_ind,:);
            z_val=z_gridvals_j(z_ind,:);            
            
%             PolicyIndexesKron_j=PolicyIndexesKron.(jstr(:));
%             d_ind=PolicyIndexesKron_j(a_ind,z_ind);
%             d_sub=ind2sub_homemade(n_d_j,d_ind);
%             d_grid_j=daz_gridstructure.d_grid.(jstr(:));
%             for kk1=1:l_d
%                 if kk1==1
%                     d_val(kk1)=d_grid_j(d_sub(kk1));
%                 else
%                     d_val(kk1)=d_grid_j(d_sub(kk1)+sum(n_d_j(1:kk1-1)));
%                 end
%             end
            az_ind=sub2ind_homemade([N_a_j,N_z_j],[a_ind,z_ind]);
            dPolicy_gridvals_j=dPolicy_gridvals.(jstr(:));
            d_val=dPolicy_gridvals_j(az_ind,:);
            
            for vv=1:length(FnsToEvaluate)
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'SSvalueParamNames={}'
                    tempcell=num2cell([d_val,a_val,z_val]');
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind);
                    tempcell=num2cell([d_val,a_val,z_val,ValuesFnParamsVec]');
                end
                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(tempcell{:});
            end
        end
    end
    SimPanelValues(:,:,ii)=SimPanelValues_ii;
end


end



