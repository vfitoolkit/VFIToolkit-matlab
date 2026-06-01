function varargout=ValueFnOnTransPath_FHorz_QuasiHyperbolic(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% Quasi-Hyperbolic discounting on FHorz transition path (compute-only, no GE).
%
% Output convention (mirrors ValueFnIter_FHorz_QuasiHyperbolic):
%   Naive:         varargout = {VPath, PolicyPath, ValtPath, PolicyaltPath}
%     VPath         = Vtilde-path      (QH-discounted value the agent sees each period)
%     PolicyPath    = QH-optimal choice (argmax of Vtilde-step)
%     ValtPath      = Valt-path        (exp-discounter continuation value carried backwards)
%     PolicyaltPath = exp-discounter argmax (needed by ValueFnFromPolicy for Naive)
%   Sophisticated: varargout = {VPath, PolicyPath, ValtPath}
%     VPath         = Vhat-path        (QH-discounted from current self's perspective)
%     PolicyPath    = equilibrium choice (argmax of Vhat-step)
%     ValtPath      = Vunderbar-path   (realised continuation under future selves' QH choices)
%
% V_final input semantics: Naive => Valt_final; Sophisticated => Vunderbar_final.
% Both are the 3rd output of ValueFnIter_Case1_FHorz with vfoptions.exoticpreferences='QuasiHyperbolic'.

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

if vfoptions.experienceasset==1
    error('ValueFnOnTransPath_FHorz_QuasiHyperbolic: experienceasset not yet supported')
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePath,ParamPath,N_j,T);

%% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
V_final=gpuArray(V_final);

%% Sizes
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
n_e=vfoptions.n_e;
N_e=prod(n_e);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

%% Set up exogenous shock processes
[z_gridvals_J, pi_z_J, ~, e_gridvals_J, pi_e_J, ~, ~, transpathoptions, vfoptions]=ExogShockSetup_FHorz_TPath(n_z,z_grid,pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,vfoptions,3);

%% Setup for V_final and Policy_final reshaping
if N_e==0
    if N_z==0
        Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_j]);
        V_final=reshape(V_final,[N_a,N_j]);
    else
        Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_z,N_j]);
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_j]);
        else
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j,N_z]);
            Policy_final=reshape(permute(Policy_final,[1,2,4,3]),[size(Policy_final,1),N_a,N_j,N_z]);
        end
    end
else
    if N_z==0
        Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_e,N_j]);
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_e,N_j]);
        else
            V_final=reshape(permute(reshape(V_final,[N_a,N_e,N_j]),[1,3,2]),[N_a*N_j,N_e]);
            Policy_final=reshape(permute(Policy_final,[1,2,4,3]),[size(Policy_final,1),N_a,N_j,N_e]);
        end
    else
        Policy_final=reshape(Policy_final,[size(Policy_final,1),N_a,N_z,N_e,N_j]);
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_e,N_j]);
        else
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_e,N_j]),[1,4,2,3]),[N_a*N_j,N_z,N_e]);
            Policy_final=reshape(permute(Policy_final,[1,2,5,3,4]),[size(Policy_final,1),N_a,N_j,N_z,N_e]);
        end
    end
end

% Policy channel count (matches Case1's convention)
PolicyChannels=l_d+l_aprime+2*(vfoptions.gridinterplayer>0);

%% Dispatch on (fastOLG, N_z, N_e)
if N_e==0
    if N_z==0
        if transpathoptions.fastOLG==0
            % fastOLG=0, no z, no e
            ValtPath=zeros(N_a,N_j,T,'gpuArray');
            ValtPath(:,:,T)=V_final;
            VPath=zeros(N_a,N_j,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_j,T,'gpuArray');
            PolicyPath(:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_j,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1 %so tt=T-ttr (ttr=0 fills slot T's perspective/Policyalt)
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,T-ttr)=Vpersp;
                if ttr>0 % preserve user's slot-T inputs for ValtPath, PolicyPath
                    ValtPath(:,:,T-ttr)=V;
                    PolicyPath(:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,T-ttr)=Policyalt;
                end
            end

        else
            % fastOLG=1, no z, no e
            ValtPath=zeros(N_a,N_j,T,'gpuArray');
            ValtPath(:,:,T)=V_final;
            VPath=zeros(N_a,N_j,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_j,T,'gpuArray');
            PolicyPath(:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_j,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,T-ttr)=V;
                    PolicyPath(:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,T-ttr)=Policyalt;
                end
            end
        end

    else % N_z>0, no e
        if transpathoptions.fastOLG==0
            % fastOLG=0, z, no e
            ValtPath=zeros(N_a,N_z,N_j,T,'gpuArray');
            ValtPath(:,:,:,T)=V_final;
            VPath=zeros(N_a,N_z,N_j,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_z,N_j,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_z,N_j,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,T-ttr)=Policyalt;
                end
            end

        else
            % fastOLG=1, z, no e
            ValtPath=zeros(N_a*N_j,N_z,T,'gpuArray');
            ValtPath(:,:,T)=V_final;
            VPath=zeros(N_a*N_j,N_z,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_j,N_z,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_j,N_z,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,T-ttr)=Policyalt;
                end
            end
        end
    end

else % N_e>0
    if N_z==0
        if transpathoptions.fastOLG==0
            % fastOLG=0, no z, e
            ValtPath=zeros(N_a,N_e,N_j,T,'gpuArray');
            ValtPath(:,:,:,T)=V_final;
            VPath=zeros(N_a,N_e,N_j,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_e,N_j,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_e,N_j,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,T-ttr)=Policyalt;
                end
            end

        else
            % fastOLG=1, no z, e
            ValtPath=zeros(N_a*N_j,N_e,T,'gpuArray');
            ValtPath(:,:,T)=V_final;
            VPath=zeros(N_a*N_j,N_e,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_j,N_e,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_j,N_e,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,T-ttr)=Policyalt;
                end
            end
        end

    else % N_z>0, N_e>0
        if transpathoptions.fastOLG==0
            % fastOLG=0, z, e
            ValtPath=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
            ValtPath(:,:,:,:,T)=V_final;
            VPath=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_z,N_e,N_j,T,'gpuArray');
            PolicyPath(:,:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_z,N_e,N_j,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,:,T-ttr)=Policyalt;
                end
            end

        else
            % fastOLG=1, z, e
            ValtPath=zeros(N_a*N_j,N_z,N_e,T,'gpuArray');
            ValtPath(:,:,:,T)=V_final;
            VPath=zeros(N_a*N_j,N_z,N_e,T,'gpuArray');
            PolicyPath=zeros(PolicyChannels,N_a,N_j,N_z,N_e,T,'gpuArray');
            PolicyPath(:,:,:,:,:,T)=Policy_final;
            if isNaive
                PolicyaltPath=zeros(PolicyChannels,N_a,N_j,N_z,N_e,T,'gpuArray');
            end

            V=V_final;
            for ttr=0:T-1
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

                if isNaive
                    [V, Policy, Policyalt, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V, Policy, Vpersp]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end

                VPath(:,:,:,T-ttr)=Vpersp;
                if ttr>0
                    ValtPath(:,:,:,T-ttr)=V;
                    PolicyPath(:,:,:,:,:,T-ttr)=Policy;
                end
                if isNaive
                    PolicyaltPath(:,:,:,:,:,T-ttr)=Policyalt;
                end
            end
        end
    end
end


%% Unkron to get into the shape for output (mirrors Case1)
if transpathoptions.fastOLG==1
    if N_e==0
        if N_z==0
            % no need to do anything
        else
            PolicyPath=permute(PolicyPath,[1,2,4,3,5]); % (daprime,a,j,z,t) -> (daprime,a,z,j,t)
            if isNaive
                PolicyaltPath=permute(PolicyaltPath,[1,2,4,3,5]);
            end
        end
    else
        if N_z==0
            PolicyPath=permute(PolicyPath,[1,2,4,3,5]);
            if isNaive
                PolicyaltPath=permute(PolicyaltPath,[1,2,4,3,5]);
            end
        else
            PolicyPath=permute(PolicyPath,[1,2,4,5,3,6]); % (daprime,a,j,z,e,t) -> (daprime,a,z,e,j,t)
            if isNaive
                PolicyaltPath=permute(PolicyaltPath,[1,2,4,5,3,6]);
            end
        end
    end
end

if N_e==0
    if N_z==0
        VPath=reshape(VPath,[n_a,N_j,T]);
        ValtPath=reshape(ValtPath,[n_a,N_j,T]);
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,N_j,T]);
        if isNaive
            PolicyaltPath=reshape(PolicyaltPath,[size(PolicyaltPath,1),n_a,N_j,T]);
        end
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,N_j,T]);
            ValtPath=reshape(ValtPath,[n_a,n_z,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,T]),[1,3,2,4]),[n_a,n_z,N_j,T]);
            ValtPath=reshape(permute(reshape(ValtPath,[N_a,N_j,N_z,T]),[1,3,2,4]),[n_a,n_z,N_j,T]);
        end
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,n_z,N_j,T]);
        if isNaive
            PolicyaltPath=reshape(PolicyaltPath,[size(PolicyaltPath,1),n_a,n_z,N_j,T]);
        end
    end
else
    if N_z==0
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_e,N_j,T]);
            ValtPath=reshape(ValtPath,[n_a,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_e,T]),[1,3,2,4]),[n_a,n_e,N_j,T]);
            ValtPath=reshape(permute(reshape(ValtPath,[N_a,N_j,N_e,T]),[1,3,2,4]),[n_a,n_e,N_j,T]);
        end
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,n_e,N_j,T]);
        if isNaive
            PolicyaltPath=reshape(PolicyaltPath,[size(PolicyaltPath,1),n_a,n_e,N_j,T]);
        end
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,n_e,N_j,T]);
            ValtPath=reshape(ValtPath,[n_a,n_z,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,N_e,T]),[1,3,4,2,5]),[n_a,n_z,n_e,N_j,T]);
            ValtPath=reshape(permute(reshape(ValtPath,[N_a,N_j,N_z,N_e,T]),[1,3,4,2,5]),[n_a,n_z,n_e,N_j,T]);
        end
        PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),n_a,n_z,n_e,N_j,T]);
        if isNaive
            PolicyaltPath=reshape(PolicyaltPath,[size(PolicyaltPath,1),n_a,n_z,n_e,N_j,T]);
        end
    end
end

%% varargout
if isNaive
    varargout={VPath,PolicyPath,ValtPath,PolicyaltPath};
else
    varargout={VPath,PolicyPath,ValtPath};
end

end
