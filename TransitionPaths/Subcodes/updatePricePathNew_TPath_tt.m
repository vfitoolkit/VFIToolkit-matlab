function [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld_tt,itercounter,transpathoptions)
% itercounter is the current shooting iteration (1 on the first), used for the additional-factor ramp
% Input size: PricePathOld_tt is 1-by-prices

p_i=zeros(1,length(GeneralEqmEqnsCell));
for gg=1:length(GeneralEqmEqnsCell)
    % Note: _v3 rather than _v3g, so on CPU rather than GPU
    p_i(gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
end

if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
    PricePathNew_tt=p_i;
    % There is no general eqm condition to report, so report the fixed-point residual instead: it is
    % zero exactly when the formulae return the price they were given, which is what convergence
    % means here. It was NaN, which never passes the convergence test on toleranceGEcondns and
    % trips the non-finite guard in the shooting loops.
    GEcondnPath_tt=PricePathNew_tt-PricePathOld_tt;
% Note there is no GEnewprice==2, it uses a completely different code
elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
    GEcondnPath_tt=p_i; % Sometimes, want to keep the GE conditions to plot them
    p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
    p_i=I_makescutoff.*p_i;
    % The additional-factor ramp (howtoupdate columns 5, 6 and 7). rampweight is 0 up to iteration
    % t1_add and 1 from iteration t2_add on, linear in between, so factor_iter is factor up to
    % t1_add, runs to f_add*factor by t2_add, and stays there. Written as 1+(f_add-1)*w rather than
    % min(...,f_add) so that f_add<1, damping the step down over iterations, works as well as
    % f_add>1. t2_add>t1_add is enforced in setupGEnewprice3_shooting, so the denominator is never
    % zero, and the max(...,0) is what holds the weight at zero before t1_add.
    rampweight=min(max((itercounter-transpathoptions.GEnewprice3.t1_add)./(transpathoptions.GEnewprice3.t2_add-transpathoptions.GEnewprice3.t1_add),0),1);
    factor_iter=transpathoptions.GEnewprice3.factor.*(1+(transpathoptions.GEnewprice3.f_add-1).*rampweight);
    % keepold is 0 exactly on the rows where the user gave factor=Inf, which means replace the price
    % outright rather than take a step from it (the setup turns that Inf into factor=1, keepold=0).
    % Matches StationaryGeneralEqm_subcode_fminalgo5 and the inline update in the GEptype branch of
    % the PType shooting codes; without it those rows kept the old price, which is the opposite of
    % what factor=Inf asks for.
    PricePathNew_tt=(PricePathOld_tt.*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*factor_iter.*p_i-(1-transpathoptions.GEnewprice3.add).*factor_iter.*p_i;
end

% We want output shapes to be
% PricePathNew_tt % output as a row vector of size 1-by-prices, because PricePath is T-by-prices
% GEcondnPath_tt  % output as a row vector of size 1-by-GECondns, because PricePath is T-by-GECondns


end