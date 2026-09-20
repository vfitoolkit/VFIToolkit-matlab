function J=TransitionPath_InfHorz_FullJacobiantriangular(p, f, nF, nPrices, PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, n_e, N_a, N_z, N_e, l_d, l_aprime, d_gridvals, aprime_gridvals, a_gridvals, a_grid, z_gridvals, e_gridvals, ze_gridvals, pi_z, pi_z_sparse, pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2)
% The brute-force Jacobian of transpathoptions.GEnewprice1.Jacobianmethod='FullJacobian', built by
% restarting the backward pass at the perturbed period instead of redoing all of it.
%
% The value fn recursion V_s = F(p_s) + beta E V_{s+1} only ever reaches forward in time, so
% perturbing the price at period tt leaves V and Policy unchanged at every period after tt. Only
% periods tt down to 1 have to be resolved, which is half the backward pass averaged over tt. This is
% exact, not an approximation: it must give the same J as the plain loop, and the test bank checks it.
%
% The forward pass gets no such saving. Perturbing the price at any period changes Policy at period 1,
% so the agent distribution differs from period 2 onwards and has to be iterated in full.
%
% The partial backward pass is done by calling TransitionPath_InfHorz_substeps_Step1_ValueFnIter
% unmodified, with T replaced by tt+1 and the terminal value fn replaced by the baseline V at tt+1.
% Every index inside that command is T-ttr, so the substitution makes it run exactly periods tt down
% to 1, read the right rows of the price and parameter paths (the rows above tt are simply never
% touched), and write only slots tt..1 of PolicyIndexesPath. Handing it a copy of the baseline
% PolicyIndexesPath therefore leaves every later period at its baseline policy, which is the point.

%% What this cannot do
% Only the case Step1_ValueFnIter actually implements. The others are error('Not yet implemented')
% there, and the baseline value fn path below would need its own loop for each of them anyway.
if ~(N_z>0 && N_e==0)
    error('transpathoptions.GEnewprice1.FullJacobianReuseVpath=1 is only implemented for models with z shocks and no e shocks. Set it to 0 to use the plain Jacobian loop')
end
if vfoptions.experienceasset>=1
    error('transpathoptions.GEnewprice1.FullJacobianReuseVpath=1 is not implemented for experience assets, which keep their own copy of the backward loops. Set it to 0 to use the plain Jacobian loop')
end

%% The baseline value fn path, which is what the perturbed passes restart from
% Step1_ValueFnIter does not return the value functions, so the baseline backward pass is done here.
% Policy is not kept: the baseline PolicyIndexesPath was already produced by the caller's own path
% solve at exactly these prices.
VPath=zeros(N_a,N_z,T,'like',V_final);
VPath(:,:,T)=V_final;
V=V_final;
Parametersbase=Parameters;
z_gridvals_tt=z_gridvals;
pi_z_tt=pi_z;
for ttr=1:T-1 % so tt=T-ttr
    for kk=1:length(PricePathNames)
        Parametersbase.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parametersbase.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if transpathoptions.zpathtrivial==0
        z_gridvals_tt=transpathoptions.z_gridvals_T(:,:,T-ttr);
        pi_z_tt=transpathoptions.pi_z_T(:,:,T-ttr);
    end
    V=ValueFnIter_InfHorz_TPath_SingleStep(V,n_d,n_a,n_z,d_gridvals, a_grid, z_gridvals_tt, pi_z_tt, ReturnFn, Parametersbase, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    VPath(:,:,T-ttr)=V;
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
        [~,PolicyIndexesPathk]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(tt+1,PolicyIndexesPathk,VPath(:,:,tt+1),Parameters,PricePathk,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals,a_grid,z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions);
        GEcondnPathk=TransitionPath_InfHorz_singlepathiter(PricePathk, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, n_e, N_a, N_z, N_e, l_d, l_aprime, d_gridvals, aprime_gridvals, a_gridvals, a_grid, z_gridvals, e_gridvals, ze_gridvals, pi_z, pi_z_sparse, pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptionsskip, itercounter, PolicyIndexesPathk, N_probs, II1, II2);
        J(:,k)=(reshape(GEcondnPathk,[],1)-f)/transpathoptions.epsprice;
    end
end

end
