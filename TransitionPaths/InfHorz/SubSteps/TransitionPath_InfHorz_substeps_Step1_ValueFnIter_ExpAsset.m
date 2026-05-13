function [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter_ExpAsset(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,N_z,N_e,d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,transpathoptions,vfoptions)
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

if N_z==0 && N_e==0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for tt=1:T-1 %so t=T-i
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        error('Not yet implemented')
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_noz(V,n_d1,n_d2,n_a1,n_a2,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,T-tt)=Policy;
    end
elseif N_z>0 && N_e==0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        if transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset(V,n_d1,n_d2,n_a1,n_a2,n_z,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    end
elseif N_z==0 && N_e>0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        if transpathoptions.zpathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr);
        end

        error('Not yet implemented')
        [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz_e(V,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    end
elseif N_z>0 && N_e>0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        if transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr);
        end

        error('Not yet implemented')
        [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_e(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    end
end


end
