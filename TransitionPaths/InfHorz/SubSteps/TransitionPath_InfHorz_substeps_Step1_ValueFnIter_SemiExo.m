function [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter_SemiExo(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals,a_grid,z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% Semi-exogenous state: transitions of semiz depend on the decision variable d2 (the last vfoptions.l_dsemiz).
% The semiz state is always present, so V always keeps a 'bothz' dimension, where bothz=(semiz,z) with semiz indexing
% fastest (when N_z=0, bothz is just semiz); e is kept separate.
%   V is [N_a,N_bothz] or [N_a,N_bothz,N_e]
%   PolicyIndexesPath is [l_d+l_a(+2),N_a,N_bothz,T-1] or [l_d+l_a(+2),N_a,N_bothz,N_e,T-1] (d1 rows, d2 rows, aprime)
% Read from vfoptions: n_semiz, setup_semiexo.semiz_gridvals [N_semiz,l_semiz], setup_semiexo.pi_semiz
% [N_semiz,N_semiz,N_d2] (gpu); and when transpathoptions.semizpathtrivial=0, transpathoptions.pi_semiz_T(:,:,:,t).
% Timing: V_t reads pi_z_T(:,:,t), pi_semiz_T(:,:,:,t) (both the t->t+1 transition), grids at t, and pi_e_T(:,t+1)
% (the distribution of next period's e).
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

n_semiz=vfoptions.n_semiz;
semiz_gridvals=vfoptions.setup_semiexo.semiz_gridvals;
pi_semiz=vfoptions.setup_semiexo.pi_semiz;

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
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_noz(V,n_d,n_a,n_semiz,d_gridvals, a_grid, semiz_gridvals, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.

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
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo(V,n_d,n_a,n_z,n_semiz,d_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
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

        % e: this period's grid, NEXT period's distribution (pi_e_T(:,t) is the distribution of the e realized IN period t)
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_noz_e(V,n_d,n_a,n_semiz,n_e,d_gridvals, a_grid, semiz_gridvals, e_gridvals, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.

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
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        % e: this period's grid, NEXT period's distribution (pi_e_T(:,t) is the distribution of the e realized IN period t)
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,T-ttr);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_e(V,n_d,n_a,n_z,n_semiz,n_e,d_gridvals, a_grid, z_gridvals, semiz_gridvals, e_gridvals, pi_z, pi_semiz, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.

        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    end
end

end
