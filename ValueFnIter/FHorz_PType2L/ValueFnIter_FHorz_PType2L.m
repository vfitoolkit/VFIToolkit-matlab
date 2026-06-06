function [V, Policy]=ValueFnIter_FHorz_PType2L(n_d,n_a,n_z, N_j, Names_i, N_i, d_grid,a_grid,z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Two-level permanent type dispatcher (top level) for finite-horizon problems.
%
% Top level is named (Names_i, a cell array). Dependence on the top level
% must be supplied as structures keyed by Names_i. Bottom level is numeric
% (N_i) and is handled by the existing ValueFnIter_Case1_FHorz_PType, which
% receives the per-top-peeled inputs and the bottom N_i.
%
% Output:
%   V.(topname).(ptypeNNN)      value function for each (top, bottom) pair
%   Policy.(topname).(ptypeNNN) policy indexes for each (top, bottom) pair
%
% vfoptions.verbose=1 will give feedback at the top-level loop
% vfoptions.verboseparams=1 will dump the per-top parameter struct

V=struct();
Policy=struct();

if ~iscell(Names_i)
    error('Names_i must be a cell array of top-level PType names for the two-level PType command.')
end
N_topi=length(Names_i);

if ~exist('vfoptions','var')
    error('You must input vfoptions; you can always set vfoptions=struct().')
end

for tt=1:N_topi
    iistr=Names_i{tt};

    % First set up vfoptions
    vfoptions_temp=PType_Options_2L(vfoptions,Names_i,tt);
    if ~isfield(vfoptions_temp,'verbose')
        vfoptions_temp.verbose=0;
    end
    if ~isfield(vfoptions_temp,'verboseparams')
        vfoptions_temp.verboseparams=0;
    end
    if ~isfield(vfoptions_temp,'ptypestorecpu')
        vfoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end

    if vfoptions_temp.verbose==1
        fprintf('Top-level permanent type: %i of %i (%s)\n',tt,N_topi,iistr)
    end

    % Go through everything which might be dependent on the top-level fixed
    % type. At this level only structure-keyed dependence is permitted; any
    % non-struct value (or struct keyed by other names, e.g. bottom-level
    % names) is passed through verbatim to the inner PType dispatcher.
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end
    if isstruct(N_i)
        N_i_temp=N_i.(iistr);
    else
        N_i_temp=N_i;
    end

    % Exogenous shocks
    [n_z_temp,z_grid_temp,pi_z_temp,vfoptions_temp]=PType_setup_ExogShocks(tt,iistr,N_topi,n_z,z_grid,pi_z,vfoptions_temp,1);

    % DiscountFactorParamNames
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isstruct(DiscountFactorParamNames)
        names=fieldnames(DiscountFactorParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},iistr)
                DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{jj});
            end
        end
    end

    % ReturnFn
    if isstruct(ReturnFn) && isfield(ReturnFn,iistr)
        ReturnFn_temp=ReturnFn.(iistr);
    else
        ReturnFn_temp=ReturnFn;
    end

    % Parameters
    Parameters_temp=PType_setup_Parameters(tt,iistr,N_topi,Parameters,1);

    if vfoptions_temp.verboseparams==1
        sprintf('Parameter values for the current top-level permanent type')
        Parameters_temp
    end

    [V_tt, Policy_tt]=ValueFnIter_Case1_FHorz_PType(n_d_temp,n_a_temp,n_z_temp,N_j_temp,N_i_temp,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp,ReturnFn_temp,Parameters_temp,DiscountFactorParamNames_temp,vfoptions_temp);

    V.(iistr)=V_tt;
    Policy.(iistr)=Policy_tt;

    clear V_tt Policy_tt

end


end
