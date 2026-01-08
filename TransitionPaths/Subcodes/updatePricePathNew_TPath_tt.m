function [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld_tt,transpathoptions)

p_i=zeros(1,length(GeneralEqmEqnsCell));
for gg=1:length(GeneralEqmEqnsCell)
    % Note: _v3 rather than _v3g, so on CPU rather than GPU
    p_i(gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters)); 
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
end

if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
    PricePathNew_tt=p_i;
    GEcondnPath_tt=[]; % not being used
% Note there is no GEnewprice==2, it uses a completely different code
elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
    GEcondnPath_tt=p_i; % Sometimes, want to keep the GE conditions to plot them
    p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
    I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
    p_i=I_makescutoff.*p_i;
    PricePathNew_tt=PricePathOld_tt+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
end


end