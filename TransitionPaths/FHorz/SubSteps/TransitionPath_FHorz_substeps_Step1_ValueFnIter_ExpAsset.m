function [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_ExpAsset(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,N_z,N_e,d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,transpathoptions,vfoptions)
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

if transpathoptions.fastOLG==0
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

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz(V,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,T-tt)=Policy;
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
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset(V,n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
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
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz_e(V,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
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
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
            end
            if transpathoptions.epathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_e(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy;
        end
    end


    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        V=V_final;
        for ttr=1:T-1 % so tt=T-ttr
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz(V,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
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
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset(V,n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_z] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z
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
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,z',z)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_e(V,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_e] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z
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
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                z_gridvals_J=transpathoptions.z_gridvals_J(:,:,T-ttr);
            end
            if transpathoptions.epathtrivial==0
                pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_e(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_z,N_e] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e
        end
    end
end
