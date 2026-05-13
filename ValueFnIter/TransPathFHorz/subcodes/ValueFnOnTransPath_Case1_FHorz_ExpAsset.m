function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz_ExpAsset(PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, Policy_final, Parameters, n_d,n_a,n_z,n_e, N_a,N_z,N_e, N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, transpathoptions, vfoptions)

if ~isfield(vfoptions,'l_dexperienceasset')
    vfoptions.l_dexperienceasset=1;
end
if length(n_d)>vfoptions.l_dexperienceasset
    n_d1=n_d(1:end-vfoptions.l_dexperienceasset);
else
    n_d1=0;
end
n_d2=n_d(end-vfoptions.l_dexperienceasset+1:end);
d2_grid=d_grid(sum(n_d1)+1:end);
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if prod(n_d1)>0
    % d1_grid=d_grid(1:sum(n_d1));
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    d_gridvals=[]; % wont be used
end
% Split endogneous states into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experienceasset
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);

a1_gridvals=CreateGridvals(n_a1,a1_grid,1);

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

if prod(n_a1)==0
    error('Not yet implemented transpath with experienceasset, without a standard endogenous state')
end

l_daprime=length(n_d)+length(n_a)-1; % remove experienceasset from l_aprime

%%
if N_e==0
    if N_z==0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, no z, no e
            VPath=zeros(N_a,N_j,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz(V,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,T-ttr)=Policy;
                VPath(:,:,T-ttr)=V;
            end
        else
            %% fastOLG=1, no z, no e

            VPath=zeros(N_a,N_j,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz(V,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,T-ttr)=Policy;
                VPath(:,:,T-ttr)=V;
            end
        end

    else % N_z>0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, z, no e
            VPath=zeros(N_a,N_z,N_j,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i


                ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset(V,n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,:,T-ttr)=Policy;
                VPath(:,:,:,T-ttr)=V;
            end

        else
            %% fastOLG=1, z, no e
            % Note: fastOLG with z: use V as (a,j)-by-z and Policy as a-by-j-by-z
            VPath=zeros(N_a*N_j,N_z,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_j,N_z,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset(V,n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,T-ttr)=V;
            end
        end
    end

else % N_e
    if N_z==0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, no z, e
            VPath=zeros(N_a,N_e,N_j,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_e,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz_e(V,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,:,T-ttr)=Policy;
                VPath(:,:,:,T-ttr)=V;
            end

        else
            %% fastOLG=1, no z, e
            % Note: fastOLG with e: use V as (a,j)-by-e and Policy as a-by-j-by-e
            VPath=zeros(N_a*N_j,N_e,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_j,N_e,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,e)
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_e(V,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,T-ttr)=V;
            end
        end
    else % N_z>0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, z, e
            VPath=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
            VPath(:,:,:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_z,N_e,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_e(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,:,:,T-ttr)=Policy;
                VPath(:,:,:,:,T-ttr)=V;
            end

        else % transpathoptions.fastOLG==1
            %% fastOLG=1, z, e
            VPath=zeros(N_a*N_j,N_z,N_e,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(l_daprime+(vfoptions.gridinterplayer>0),N_a,N_j,N_z,N_e,T,'gpuArray');
            PolicyPath(:,:,:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_e(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e, N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,:,T-ttr)=V;
            end
        end
    end
end



end
