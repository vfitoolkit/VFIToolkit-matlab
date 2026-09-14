function vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions)

[ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8, sj, warmglow] = ...
    EpsteinZinSetup_FHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);

% --- NEW: PACK EZ CONSTANTS FOR THE UNIVERSAL ORCHESTRATOR ---
vfoptions.ezc2 = ezc2;
vfoptions.ezc3 = ezc3;
vfoptions.ezc4 = ezc4;
vfoptions.ezc5 = ezc5;
vfoptions.ezc6 = ezc6;
vfoptions.ezc7 = ezc7;
vfoptions.ezc8 = ezc8;
vfoptions.sj = sj;
vfoptions.warmglow = warmglow;


end
