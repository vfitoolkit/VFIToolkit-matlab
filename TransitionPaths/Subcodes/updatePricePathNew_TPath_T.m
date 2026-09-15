function PricePathNew=updatePricePathNew_TPath_T(GEcondnPath,PricePathOld,T,itercounter,transpathoptions)
% Whole-path version of updatePricePathNew_TPath_tt: takes the general eqm conditions for every period
% and returns the whole new price path. The shooting algorithm updates each price from its own
% condition in its own period and so does not need the other periods, but the Newton options do, as
% their Jacobian couples periods. Used when transpathoptions.updatepert=0.
%
% Input sizes: GEcondnPath is (T-1)-by-GEcondns, PricePathOld is T-by-prices
% itercounter is the current iteration (1 on the first), used for the additional-factor ramp

PricePathNew=zeros(size(PricePathOld),'like',PricePathOld);
PricePathNew(T,:)=PricePathOld(T,:); % period T is the terminal condition, it never updates

if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
    PricePathNew(1:T-1,:)=GEcondnPath; % the formulae evaluate to the new prices directly
elseif transpathoptions.GEnewprice==2 % Anderson acceleration, which accelerates the shooting update, so the map itself is the shooting one
    % Same formula as GEnewprice==3 below, but with no additional-factor ramp: Anderson builds its
    % step from the history of iterates and so needs the same map at every iteration, which is why
    % setupGEnewprice3_shooting errors if additionalfactor is set for GEnewprice=2.
    p_i=GEcondnPath(:,transpathoptions.GEnewprice2.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    p_i=(abs(p_i)>transpathoptions.updateaccuracycutoff).*p_i;
    PricePathNew(1:T-1,:)=(PricePathOld(1:T-1,:).*transpathoptions.GEnewprice2.keepold)+transpathoptions.GEnewprice2.add.*transpathoptions.GEnewprice2.factor.*p_i-(1-transpathoptions.GEnewprice2.add).*transpathoptions.GEnewprice2.factor.*p_i;
elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
    p_i=GEcondnPath(:,transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
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
    % add, factor_iter and keepold are all 1-by-prices, so they apply to every period.
    PricePathNew(1:T-1,:)=(PricePathOld(1:T-1,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*factor_iter.*p_i-(1-transpathoptions.GEnewprice3.add).*factor_iter.*p_i;
end

end
