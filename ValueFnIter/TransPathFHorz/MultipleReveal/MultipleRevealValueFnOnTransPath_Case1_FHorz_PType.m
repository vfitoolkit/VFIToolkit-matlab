function [RealizedVPath, RealizedPolicyPath, VPath, PolicyPath]=MultipleRevealValueFnOnTransPath_Case1_FHorz_PType(PricePath, ParamPath, T, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions_path, vfoptions_finaleqm)

revealperiodnames=fieldnames(ParamPath);
nReveals=length(revealperiodnames);
% Just assume that the reveals are all set up correctly, as they already solved PricePath
revealperiods=zeros(nReveals,1);
for rr=1:nReveals
    currentrevealname=revealperiodnames{rr};
    try
        revealperiods(rr)=str2double(currentrevealname(2:end));
    catch
        error('Multiple reveal transition paths: a field in ParamPath is misnamed (it must be tXXXX, where the X are numbers; you have one of the XXXX not being a number)')
    end
end
historylength=revealperiods(nReveals)+T-1; % length of realized path

% Setup N_i and Names_i
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
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

for rr=1:nReveals
    PricePath_rr=PricePath.(revealperiodnames{rr});
    ParamPath_rr=ParamPath.(revealperiodnames{rr});

    if rr<nReveals
        durationofreveal_rr=revealperiods(rr+1)-revealperiods(rr);
    else
        durationofreveal_rr=T;
    end

    % First we need V_final (it is assumed that the final period of
    % PricePath_rr contains the stationary general eqm; it should)
    PricePathNames=fieldnames(PricePath_rr);
    for pp=1:length(PricePathNames)
        temp=PricePath_rr.(PricePathNames{pp});
        Parameters.(PricePathNames{pp})=temp(end); % this cannot yet handle age-dependent params
    end
    ParamPathNames=fieldnames(ParamPath_rr);
    for pp=1:length(ParamPathNames)
        temp=ParamPath_rr.(ParamPathNames{pp});
        Parameters.(ParamPathNames{pp})=temp(end); % this cannot yet handle age-dependent params
    end
    [V_final,Policy_final]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid,pi_z,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions_finaleqm);

    % Now compute the paths
    [VPath_rr,PolicyPath_rr]=ValueFnOnTransPath_Case1_FHorz_PType(PricePath_rr, ParamPath_rr, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions_path);
    VPath.(revealperiodnames{rr})=VPath_rr;
    PolicyPath.(revealperiodnames{rr})=PolicyPath_rr;

    for ii=1:N_i
        if rr==1
            temp_vsize=size(VPath_rr.(Names_i{ii}));
            RealizedVPath.(Names_i{ii})=zeros([prod(temp_vsize(1:end-1)),historylength]);
            temp_policysize=size(PolicyPath_rr.(Names_i{ii}));
            RealizedPolicyPath.(Names_i{ii})=zeros([prod(temp_policysize(1:end-1)),historylength]);
        end
        
        temp=RealizedVPath.(Names_i{ii});
        temp2=reshape(VPath_rr.(Names_i{ii}),[prod(temp_vsize(1:end-1)),T]);
        if rr<nReveals
            temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal_rr);
        else
            temp(:,revealperiods(rr):end)=temp2(:,1:durationofreveal_rr);
        end
        RealizedVPath.(Names_i{ii})=temp;

        temp=RealizedPolicyPath.(Names_i{ii});
        temp2=reshape(PolicyPath_rr.(Names_i{ii}),[prod(temp_policysize(1:end-1)),T]);
        if rr<nReveals
            temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal_rr);
        else
            temp(:,revealperiods(rr):end)=temp2(:,1:durationofreveal_rr);            
        end
        RealizedPolicyPath.(Names_i{ii})=temp;
    end
end

% Reshape for output (get them out of kron from)
for ii=1:N_i
    RealizedVPath.(Names_i{ii})=reshape(RealizedVPath.(Names_i{ii}),[temp_vsize(1:end-1),historylength]);
    RealizedPolicyPath.(Names_i{ii})=reshape(RealizedPolicyPath.(Names_i{ii}),[temp_policysize(1:end-1),historylength]);
end

end


