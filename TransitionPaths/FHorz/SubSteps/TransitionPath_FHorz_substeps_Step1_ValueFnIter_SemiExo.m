function [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_SemiExo(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d1,n_d2,n_a,n_semiz,n_z,n_e,N_j,N_z,N_e,d1_gridvals,d2_gridvals,a_grid,z_gridvals_J,semiz_gridvals_J,e_gridvals_J,pi_z_J,pi_semiz_J,pi_e_J,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% Semi-exogenous state: transitions of semiz depend on the decision variable d2.
% The semiz state is always present, so V always keeps a 'bothz' dimension, where bothz=(semiz,z) with semiz indexing fastest (when N_z=0, bothz is just semiz).
% Inputs:
%   pi_semiz_J is (semiz,semiz',d2,j) [standard form, both for fastOLG=0 and =1]
%   semiz_gridvals_J is (N_semiz,l_semiz,N_j) if transpathoptions.fastOLG=0, and (N_j,N_semiz,l_semiz) if transpathoptions.fastOLG=1
%   z_gridvals_J/pi_z_J/e_gridvals_J/pi_e_J are in the standard ExogShockSetup_FHorz_TPath formats for the relevant fastOLG
%   [Note: for fastOLG=1 with e, pi_e_J must be (N_a*N_j,1,N_e), including when N_z=0 (because semiz keeps the bothz dimension in V)]
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
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

            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz(V,n_d1,n_d2,n_a,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
        end
    elseif N_z>0 && N_e==0
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
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo(V,n_d1,n_d2,n_a,n_z,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
        end
    elseif N_z==0 && N_e>0
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.epathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz_e(V,n_d1,n_d2,n_a,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy;
        end
    elseif N_z>0 && N_e>0
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
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_e(V,n_d1,n_d2,n_a,n_z,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

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

            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_noz(V,n_d1,n_d2,n_a,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-semiz
        end
    elseif N_z>0 && N_e==0
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
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo(V,n_d1,n_d2,n_a,n_z,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-bothz
        end
    elseif N_z==0 && N_e>0
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.epathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=reshape(transpathoptions.pi_e_J_T(:,:,T-ttr),[prod(n_a)*N_j,1,prod(n_e)]); % semiz keeps the bothz dimension in V, so need pi_e_J as (a,j)-by-1-by-e
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_noz_e(V,n_d1,n_d2,n_a,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-semiz-by-e
        end
    elseif N_z>0 && N_e>0
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
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
            end
            if transpathoptions.epathtrivial==0
                pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_e(V,n_d1,n_d2,n_a,n_z,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The V input is next period value fn, the V output is this period.

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-bothz-by-e
        end
    end
end

end
