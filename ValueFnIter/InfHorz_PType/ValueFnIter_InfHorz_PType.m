function [V, Policy]=ValueFnIter_InfHorz_PType(n_d,n_a,n_z,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

%
% vfoptions.verbose=1 will give feedback
% vfoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
% N_i=prod(n_i);

if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

for ii=1:N_i
    iistr=Names_i{ii};

    % First set up vfoptions
    if exist('vfoptions','var')
        vfoptions_temp=PType_Options(vfoptions,iistr);
        if ~isfield(vfoptions_temp,'verbose')
            vfoptions_temp.verbose=0;
        end
        if ~isfield(vfoptions_temp,'verboseparams')
            vfoptions_temp.verboseparams=0;
        end
        if ~isfield(vfoptions_temp,'ptypestorecpu')
            vfoptions_temp.ptypestorecpu=0; % =1 can be used as GPU memory is limited, so switch solutions to the cpu
        end
    else
        vfoptions_temp.verbose=0;
        vfoptions_temp.verboseparams=0;
        vfoptions_temp.ptypestorecpu=0; % =1 can be used as GPU memory is limited, so switch solutions to the cpu
    end

    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,pi_z_temp,vfoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,vfoptions_temp,3);

    % ReturnFn and DiscountFactor
    [ReturnFn_temp, DiscountFactorParamNames_temp]=PType_setup_ReturnFnDiscountFactor(iistr,ReturnFn,DiscountFactorParamNames);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if vfoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end

    [V_ii, Policy_ii]=ValueFnIter_InfHorz(n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, [], vfoptions_temp);

    if vfoptions_temp.ptypestorecpu==1
        V.(iistr)=gather(V_ii);
        Policy.(iistr)=gather(Policy_ii);
    else
        V.(iistr)=V_ii;
        Policy.(iistr)=Policy_ii;
    end

    clear V_ii Policy_ii

end


end
