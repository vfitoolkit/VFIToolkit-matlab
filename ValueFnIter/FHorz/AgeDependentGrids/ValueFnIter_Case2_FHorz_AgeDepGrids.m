function [V, Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)
% Do not call this command directly, it is a subcommand of ValueFnIter_Case2_FHorz()

%%
daz_gridstructure=AgeDependentGrids_Create_daz_gridstructure(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Parameters, vfoptions);
% Creates daz_gridstructure which contains both the grids themselves and a
% bunch of info about the grids in an easy to access way.
% e.g., the d_grid for age j=10: daz_gridstructure.d_grid.j010
% e.g., the value of N_a for age j=5: daz_gridstructure.N_a.j005
% e.g., the zprime_grid for age j=20: daz_gridstructure.zprime_grid.j020

%%
if vfoptions.verbose==1
    vfoptions
end

% EXOTIC PREFERENCES NOT YET IMPLEMENTED

%% Deal with Dynasty_CareAboutDecendents if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    
    if vfoptions.nphi==1
        if vfoptions.parallel==0
            disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
%             [V,Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids_Dynasty_raw(daz_gridstructure,N_j,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        elseif vfoptions.parallel==2
            [V,Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids_Dynasty_Par2_raw(daz_gridstructure,N_j,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
        end
%     else
    end
    
    % Transform V & PolicyIndexes out of kroneckered form
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
        
        V.(jstr)=reshape(V.(jstr),[n_a_j,n_z_j]);
        Policy.(jstr)=UnKronPolicyIndexes_Case2(Policy.(jstr), n_d_j, n_a_j, n_z_j,vfoptions); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
    end
    
    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        Policy=uint64(Policy);
        Policy=double(Policy);
    end
    
    return
    
end

%% 
if vfoptions.nphi==1
    if vfoptions.parallel==0
        disp('WARNING: FINITE HORZ VALUEFNITER CODES ONLY REALLY WORK ON GPU (PARALLEL=2)')
%        [VKron,PolicyKron]=ValueFnIter_Case2_FHorz_AgeDependentGrids_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    elseif vfoptions.parallel==2
        [V,Policy]=ValueFnIter_Case2_FHorz_AgeDepGrids_Par2_raw(daz_gridstructure,N_j,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    end
% else
end

% Transform V & PolicyIndexes out of kroneckered form
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

    V.(jstr)=reshape(V.(jstr),[n_a_j,n_z_j]);
    Policy.(jstr)=UnKronPolicyIndexes_Case2(Policy.(jstr), n_d_j, n_a_j, n_z_j,vfoptions); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
end
    
% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end

end
