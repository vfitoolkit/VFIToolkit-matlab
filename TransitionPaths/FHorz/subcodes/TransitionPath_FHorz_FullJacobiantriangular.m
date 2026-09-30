function J=TransitionPath_FHorz_FullJacobiantriangular(p, f, nF, nPrices, PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj)
% The brute-force Jacobian of transpathoptions.GEnewprice1.Jacobianmethod='FullJacobian', built by
% restarting the backward pass at the perturbed period instead of redoing all of it.
% The FHorz version of TransitionPath_InfHorz_FullJacobiantriangular.
%
% The value fn recursion V_s = F(p_s) + beta E V_{s+1} only ever reaches forward in time, so
% perturbing the price at period tt leaves V and Policy unchanged at every period after tt. Only
% periods tt down to 1 have to be resolved, which is half the backward pass averaged over tt. This is
% exact, not an approximation: it must give the same J as the plain loop, and the test bank checks it.
% Here V is the whole age profile at each period, (a,z,j) or (a,j,z) under fastOLG, but that changes
% nothing about the argument, since period tt still only reads period tt+1.
%
% The forward pass gets no such saving. Perturbing the price at any period changes Policy at period 1,
% so the agent distribution differs from period 2 onwards and has to be iterated in full.
%
% The partial backward pass is done by calling TransitionPath_FHorz_substeps_Step1_ValueFnIter
% unmodified, with T replaced by tt+1 and the terminal value fn replaced by the baseline V at tt+1.
% Every index inside that command is T-ttr, so the substitution makes it run exactly periods tt down
% to 1, read the right rows of the price and parameter paths (the rows above tt are simply never
% touched), and write only slots tt..1 of PolicyIndexesPath. Handing it a copy of the baseline
% PolicyIndexesPath therefore leaves every later period at its baseline policy, which is the point.

%% What this cannot do
% Only the standard endogenous state with z and no e, the same scope as the InfHorz version. The
% baseline value fn path below would need its own loop for each of the others.
if ~(N_z>0 && N_e==0 && N_semiz==0)
    error('transpathoptions.GEnewprice1.FullJacobianReuseVpath=1 is only implemented for models with z shocks and no e or semi-exogenous shocks. Set it to 0 to use the plain Jacobian loop')
end
if vfoptions.experienceasset>=1 || vfoptions.experienceassetz>=1
    error('transpathoptions.GEnewprice1.FullJacobianReuseVpath=1 is not implemented for experience assets, which keep their own copy of the backward loops. Set it to 0 to use the plain Jacobian loop')
end
if ~strcmp(vfoptions.exoticpreferences,'None')
    error('transpathoptions.GEnewprice1.FullJacobianReuseVpath=1 is not implemented for exotic preferences, which keep their own copy of the backward loops. Set it to 0 to use the plain Jacobian loop')
end

%% The baseline value fn path, which is what the perturbed passes restart from
% Step1_ValueFnIter does not return the value functions, so the baseline backward pass is done here,
% with exactly the single-step commands and the same period-by-period Parameters that Step1 uses.
% Policy is not kept: the baseline PolicyIndexesPath was already produced by the caller's own path
% solve at exactly these prices.
Parametersbase=Parameters;
z_gridvals_J_tt=z_gridvals_J;
pi_z_J_tt=pi_z_J;
if transpathoptions.fastOLG==0
    VPath=zeros(N_a,N_z,N_j,T,'like',V_final); % V_final is [N_a,N_z,N_j]
    VPath(:,:,:,T)=V_final;
else
    VPath=zeros(N_a*N_j,N_z,T,'like',V_final); % fastOLG: V_final is [N_a*N_j,N_z]
    VPath(:,:,T)=V_final;
end
V=V_final;
for ttr=1:T-1 % so tt=T-ttr
    for kk=1:length(PricePathNames)
        Parametersbase.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parametersbase.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if transpathoptions.zpathtrivial==0
        z_gridvals_J_tt=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
        pi_z_J_tt=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
    end
    if transpathoptions.fastOLG==0
        V=ValueFnIter_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J_tt, pi_z_J_tt, ReturnFn, Parametersbase, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        VPath(:,:,:,T-ttr)=reshape(V,[N_a,N_z,N_j]);
    else
        V=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J_tt, pi_z_J_tt, ReturnFn, Parametersbase, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        VPath(:,:,T-ttr)=reshape(V,[N_a*N_j,N_z]); % the raws do not all return V in the same one of the two (a,j,z) shapes, so store it in V_final's
    end
end

%% One column per price per period
% p is stacked price-major, so column k of the Jacobian is price ip in period tt
transpathoptionsskip=transpathoptions;
transpathoptionsskip.skipStep1ValueFnIter=1; % PolicyIndexesPathk below is already solved
J=zeros(nF,nF,'gpuArray');
for ip=1:nPrices
    for tt=1:T-1
        k=(ip-1)*(T-1)+tt;
        pk=p;
        pk(k)=pk(k)+transpathoptions.epsprice;
        PricePathk=PricePathOld;
        PricePathk(1:T-1,:)=reshape(pk,T-1,nPrices);
        PolicyIndexesPathk=PolicyIndexesPath; % the baseline, kept at every period after tt
        if transpathoptions.fastOLG==0
            V_ttplus1=VPath(:,:,:,tt+1);
        else
            V_ttplus1=VPath(:,:,tt+1);
        end
        [~,PolicyIndexesPathk]=TransitionPath_FHorz_substeps_Step1_ValueFnIter(tt+1,PolicyIndexesPathk,V_ttplus1,Parameters,PricePathk,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);
        GEcondnPathk=TransitionPath_FHorz_singlepathiter(PricePathk, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptionsskip, itercounter, PolicyIndexesPathk, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
        J(:,k)=(reshape(GEcondnPathk,[],1)-f)/transpathoptions.epsprice;
    end
end

end
