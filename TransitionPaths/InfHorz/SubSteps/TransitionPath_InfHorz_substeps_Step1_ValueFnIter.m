function [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals, a_grid, z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

if vfoptions.experienceasset==1
    [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter_ExpAsset(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,vfoptions.setup_experienceasset.n_d1,vfoptions.setup_experienceasset.n_d2,vfoptions.setup_experienceasset.n_a1,vfoptions.setup_experienceasset.n_a2,n_z,n_e,N_z,N_e,d_gridvals, vfoptions.setup_experienceasset.d2_gridvals,vfoptions.setup_experienceasset.a1_gridvals,vfoptions.setup_experienceasset.a2_grid, z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,vfoptions.setup_experienceasset.aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,vfoptions.setup_experienceasset.aprimeFnParamNames,transpathoptions,vfoptions);
    return
end

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
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz(V,n_d,n_a,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep(V,n_d,n_a,n_z,d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

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
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e(V,n_d,n_a,n_e,d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    end
end


