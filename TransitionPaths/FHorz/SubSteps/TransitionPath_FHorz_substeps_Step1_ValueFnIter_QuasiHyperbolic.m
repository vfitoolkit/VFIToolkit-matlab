function [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_QuasiHyperbolic(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% Step1 backwards-VFI sweep over path periods for QuasiHyperbolic discounting.
%
% Carry-state V (and V_final input) is Valt for Naive / Vunderbar for Sophisticated.
% PolicyIndexesPath stores the agent's actual policy (QH-optimal for Naive,
% equilibrium for Sophisticated) - same shape as the standard Step1 output.
%
% Naive's exp-discounter argmax (Policyalt) is computed inside the QH single-step
% wrapper but discarded here - the shooting algorithm only needs the actual policy
% for forward agent-dist iteration. ValueFnOnTransPath_FHorz_QuasiHyperbolic
% handles Policyalt-Path separately when callers request it.

VPath=[];

if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
        V=V_final;
        for tt=1:T-1 %so t=T-i
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz(V,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,T-tt)=Policy;
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

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

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

            if transpathoptions.zpathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz_e(V,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
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

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_e(V,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy;
        end
    end


    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        V=V_final;
        for ttr=1:T-1 % so tt=T-ttr
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz(V,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
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

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z
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

            if transpathoptions.zpathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,z',z)
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
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
                z_gridvals_J=transpathoptions.z_gridvals_J(:,:,T-ttr);
            end
            if transpathoptions.epathtrivial==0
                pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e
        end
    end
end

end
