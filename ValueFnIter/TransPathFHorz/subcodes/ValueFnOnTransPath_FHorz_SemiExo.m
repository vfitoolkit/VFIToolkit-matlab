function [VPath,PolicyPath]=ValueFnOnTransPath_FHorz_SemiExo(PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, Policy_final, Parameters, n_d,n_a,n_z,n_e, N_a,N_z,N_e, N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, transpathoptions, vfoptions)
% Semi-exogenous state: transitions of semiz depend on the decision variable d2 (the last vfoptions.l_dsemiz decision variables).
% Note: unlike the ExpAsset/ExpAssetz versions, this is called BEFORE the standard reshapes of V_final/Policy_final in
% ValueFnOnTransPath_Case1_FHorz (because those reshapes do not know about the semiz dimension). So V_final and
% Policy_final arrive in their user shapes, and this command does all the (un)reshaping itself.
% Internally semiz and z are treated as the single composite bothz=(semiz,z), with semiz indexing fastest.

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);
N_bothz=N_semiz*max(N_z,1); % if no z, bothz is just semiz
if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end

if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end


%% Split the decision variables into d1 (standard) and d2 (those influencing the semi-exogenous state)
if ~isfield(vfoptions,'l_dsemiz')
    vfoptions.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end
if length(n_d)>vfoptions.l_dsemiz
    n_d1=n_d(1:end-vfoptions.l_dsemiz);
    d1_grid=d_grid(1:sum(n_d1));
else
    n_d1=0; d1_grid=[];
end
n_d2=n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
d2_grid=d_grid(sum(n_d1)+1:end);

d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

l_d=length(n_d);
l_aprime=1; % semiz only allows scalar n_a

%% Set up the semi-exogenous state
% Check whether any of the parameters of SemiExoStateFn are on the path (in which case pi_semiz_J varies over the transition)
transpathoptions.semizpathtrivial=1;
temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
nargsSemiExo=2*length(n_semiz)+vfoptions.l_dsemiz; % first inputs are (semiz,semizprime,dsemiz)
if length(temp)>nargsSemiExo
    SemiExoStateFnParamNames={temp{nargsSemiExo+1:end}};
else
    SemiExoStateFnParamNames={};
end
for kk=1:length(SemiExoStateFnParamNames)
    if any(strcmp(ParamPathNames,SemiExoStateFnParamNames{kk})) || any(strcmp(PricePathNames,SemiExoStateFnParamNames{kk}))
        transpathoptions.semizpathtrivial=0;
    end
end
if transpathoptions.semizpathtrivial==0
    error('Parameters of vfoptions.SemiExoStateFn appearing on PricePath/ParamPath are not yet implemented for transition paths (the semi-exogenous transition probabilities would need to vary over the transition path) - email me if you want this')
end

vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
% output: vfoptions.semiz_gridvals_J [N_semiz,l_semiz,N_j], vfoptions.pi_semiz_J [N_semiz,N_semiz',N_d2,N_j]
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;
if transpathoptions.fastOLG==1
    semiz_gridvals_J=permute(semiz_gridvals_J,[3,1,2]); % fastOLG form: (N_j,N_semiz,l_semiz)
    % pi_semiz_J stays in standard form (semiz,semiz',d2,j); the fastOLG codes handle it internally
end


%% For fastOLG with e but no z, semiz means V keeps a bothz dimension, so we need the 'with z' fastOLG format for pi_e_J
if transpathoptions.fastOLG==1 && N_e>0 && N_z==0
    pi_e_J=reshape(pi_e_J,[N_a*N_j,1,N_e]); % was (N_a*N_j,N_e) from ExogShockSetup_FHorz_TPath with N_z=0
end

%% Reshape V_final and Policy_final (the semiz analogue of the standard reshapes in ValueFnOnTransPath_Case1_FHorz)
% Note: I keep Policy as having a first dimension (even if it is just 1)
if N_e==0
    Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_bothz,N_j]);
    if transpathoptions.fastOLG==0
        V_final=reshape(V_final,[N_a,N_bothz,N_j]);
    else % transpathoptions.fastOLG==1
        V_final=reshape(permute(reshape(V_final,[N_a,N_bothz,N_j]),[1,3,2]),[N_a*N_j,N_bothz]);
        Policy_final=reshape(permute(Policy_final,[1,2,4,3]),[size(Policy_final,1),N_a,N_j,N_bothz]);
    end
else
    Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_bothz,N_e,N_j]);
    if transpathoptions.fastOLG==0
        V_final=reshape(V_final,[N_a,N_bothz,N_e,N_j]);
    else % transpathoptions.fastOLG==1
        V_final=reshape(permute(reshape(V_final,[N_a,N_bothz,N_e,N_j]),[1,4,2,3]),[N_a*N_j,N_bothz,N_e]);
        Policy_final=reshape(permute(Policy_final,[1,2,5,3,4]),[size(Policy_final,1),N_a,N_j,N_bothz,N_e]);
    end
end

%%
if N_e==0
    if transpathoptions.fastOLG==0
        %% fastOLG=0, no e
        VPath=zeros(N_a,N_bothz,N_j,T,'gpuArray');
        VPath(:,:,:,T)=V_final;
        PolicyPath=zeros(l_d+l_aprime+2*(vfoptions.gridinterplayer>0),N_a,N_bothz,N_j,T,'gpuArray'); %Periods 1 to T-1
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

            if N_z>0
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            if N_z==0
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz(V,n_d1,n_d2,n_a,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo(V,n_d1,n_d2,n_a,n_z,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            % The V input is next period value fn, the V output is this period.

            PolicyPath(:,:,:,:,T-ttr)=Policy;
            VPath(:,:,:,T-ttr)=V;
        end

    else
        %% fastOLG=1, no e
        % Note: fastOLG: use V as (a,j)-by-bothz and Policy as a-by-j-by-bothz
        VPath=zeros(N_a*N_j,N_bothz,T,'gpuArray');
        VPath(:,:,T)=V_final;
        PolicyPath=zeros(l_d+l_aprime+2*(vfoptions.gridinterplayer>0),N_a,N_j,N_bothz,T,'gpuArray');
        PolicyPath(:,:,:,:,T)=Policy_final;

        % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if N_z>0
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            if N_z==0
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_noz(V,n_d1,n_d2,n_a,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo(V,n_d1,n_d2,n_a,n_z,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            % The V input is next period value fn, the V output is this period.

            PolicyPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-bothz
            VPath(:,:,T-ttr)=V;
        end
    end

else % N_e>0
    if transpathoptions.fastOLG==0
        %% fastOLG=0, e
        VPath=zeros(N_a,N_bothz,N_e,N_j,T,'gpuArray');
        VPath(:,:,:,:,T)=V_final;
        PolicyPath=zeros(l_d+l_aprime+2*(vfoptions.gridinterplayer>0),N_a,N_bothz,N_e,N_j,T,'gpuArray'); %Periods 1 to T-1
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
            if N_z>0
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr);
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr);
            end

            if N_z==0
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz_e(V,n_d1,n_d2,n_a,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_e(V,n_d1,n_d2,n_a,n_z,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            % The V input is next period value fn, the V output is this period.

            PolicyPath(:,:,:,:,:,T-ttr)=Policy;
            VPath(:,:,:,:,T-ttr)=V;
        end

    else % transpathoptions.fastOLG==1
        %% fastOLG=1, e
        VPath=zeros(N_a*N_j,N_bothz,N_e,T,'gpuArray');
        VPath(:,:,:,T)=V_final;
        PolicyPath=zeros(l_d+l_aprime+2*(vfoptions.gridinterplayer>0),N_a,N_j,N_bothz,N_e,T,'gpuArray');
        PolicyPath(:,:,:,:,:,T)=Policy_final;

        % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.epathtrivial==0
                if N_z==0
                    pi_e_J=reshape(transpathoptions.pi_e_J_T(:,:,T-ttr),[N_a*N_j,1,N_e]); % semiz keeps the bothz dimension in V, so need pi_e_J as (a,j)-by-1-by-e
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                else
                    pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
                end
            end
            if N_z>0
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end
            end
            if transpathoptions.semizpathtrivial==0
                semiz_gridvals_J=transpathoptions.semiz_gridvals_J_T(:,:,:,T-ttr); % fastOLG: (N_j,N_semiz,l_semiz)
                pi_semiz_J=transpathoptions.pi_semiz_J_T(:,:,:,:,T-ttr); % standard form (semiz,semiz',d2,j)
            end

            if N_z==0
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_noz_e(V,n_d1,n_d2,n_a,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_e(V,n_d1,n_d2,n_a,n_z,n_semiz,n_e,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            % The V input is next period value fn, the V output is this period.

            PolicyPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-bothz-by-e
            VPath(:,:,:,T-ttr)=V;
        end
    end
end


%% Unkron to get into the shape for output
if transpathoptions.fastOLG==1
    % fastOLG, so need to permute back to standard ordering
    if N_e==0
        PolicyPath=permute(PolicyPath,[1,2,4,3,5]); % was (daprime,a,j,bothz,t), now (daprime,a,bothz,j,t)
    else
        PolicyPath=permute(PolicyPath,[1,2,4,5,3,6]); % was (daprime,a,j,bothz,e,t), now (daprime,a,bothz,e,j,t)
    end
end

% Then the unkron itself
if N_e==0
    if transpathoptions.fastOLG==0
        VPath=reshape(VPath,[n_a,n_bothz,N_j,T]);
    else
        VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_bothz,T]),[1,3,2,4]),[n_a,n_bothz,N_j,T]);
    end
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,n_bothz,N_j,T]);
else
    if transpathoptions.fastOLG==0
        VPath=reshape(VPath,[n_a,n_bothz,n_e,N_j,T]);
    else
        VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_bothz,N_e,T]),[1,3,4,2,5]),[n_a,n_bothz,n_e,N_j,T]);
    end
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,n_bothz,n_e,N_j,T]);
end


end
