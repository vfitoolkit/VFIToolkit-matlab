function [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_PTypeDependent_tt(Parameters,Parameters_ii,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,Names_i,N_i,PricePathOld_tt,itercounter,transpathoptions)
% The permanent-type version of updatePricePathNew_TPath_tt, for when at least one general eqm
% condition is declared to depend on permanent type (transpathoptions.GEptype). Such a condition is
% evaluated once per permanent type, against that type's own parameters, and each of those gets its
% own price to clear. So p_i has sum(GEptype==0)+N_i*sum(GEptype==1) entries rather than one per
% general eqm eqn, which is why this cannot just call updatePricePathNew_TPath_tt.
%
% Input size: PricePathOld_tt is 1-by-prices
% itercounter is the current shooting iteration (1 on the first), used for the additional-factor ramp
%
% Parameters    holds the values common to all permanent types
% Parameters_ii is a struct with one field per permanent type, named by Names_i

nGeneralEqmEqns=length(GeneralEqmEqnsCell);

p_i=zeros(1,sum(transpathoptions.GEptype==0)+N_i*sum(transpathoptions.GEptype==1));
gg_c=0;
for gg=1:nGeneralEqmEqns
    if transpathoptions.GEptype(gg)==0
        gg_c=gg_c+1;
        p_i(gg_c)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
    elseif transpathoptions.GEptype(gg)==1
        for ii=1:N_i
            iistr=Names_i{ii};
            gg_c=gg_c+1;
            p_i(gg_c)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_ii.(iistr)));
        end
    end
end
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex

if transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
    GEcondnPath_tt=p_i; % Sometimes, want to keep the GE conditions to plot them
    p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
    p_i=I_makescutoff.*p_i;
    % The additional-factor ramp (howtoupdate columns 5, 6 and 7). rampweight is 0 up to iteration
    % t1_add and 1 from iteration t2_add on, linear in between, so factor_iter is factor up to
    % t1_add, runs to f_add*factor by t2_add, and stays there. t2_add>t1_add is enforced in
    % setupGEnewprice3_shooting, so the denominator is never zero.
    rampweight=min(max((itercounter-transpathoptions.GEnewprice3.t1_add)./(transpathoptions.GEnewprice3.t2_add-transpathoptions.GEnewprice3.t1_add),0),1);
    factor_iter=transpathoptions.GEnewprice3.factor.*(1+(transpathoptions.GEnewprice3.f_add-1).*rampweight);
    % keepold is 0 exactly on the rows where the user gave factor=Inf, which means replace the price
    % outright rather than take a step from it (the setup turns that Inf into factor=1, keepold=0).
    PricePathNew_tt=(PricePathOld_tt.*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*factor_iter.*p_i-(1-transpathoptions.GEnewprice3.add).*factor_iter.*p_i;
else
    % The PType solvers only accept GEnewprice=3 here (GEnewprice=2 accelerates the same update from outside it)
    error('updatePricePathNew_TPath_PTypeDependent_tt: only transpathoptions.GEnewprice=3 is implemented (got %g)',transpathoptions.GEnewprice)
end

% We want output shapes to be
% PricePathNew_tt % output as a row vector of size 1-by-prices, because PricePath is T-by-prices
% GEcondnPath_tt  % output as a row vector of size 1-by-GECondns, because PricePath is T-by-GECondns


end
