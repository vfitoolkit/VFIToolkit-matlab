function [RealizedAggVarsPath, AggVarsPath]=MultipleRevealEvalFnOnTransPath_AggVars_Case1_FHorz_PType(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, transpathoptions, simoptions)

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

AggVarNames=fieldnames(FnsToEvaluate);
for rr=1:nReveals
    PricePath_rr=PricePath.(revealperiodnames{rr});
    ParamPath_rr=ParamPath.(revealperiodnames{rr});
    AgentDistPath_rr=AgentDistPath.(revealperiodnames{rr});
    PolicyPath_rr=PolicyPath.(revealperiodnames{rr});

    if rr<nReveals
        durationofreveal_rr=revealperiods(rr+1)-revealperiods(rr);
    else
        durationofreveal_rr=T;
    end

    AggVarsPath_rr=EvalFnOnTransPath_AggVars_Case1_FHorz_PType(FnsToEvaluate, AgentDistPath_rr, PolicyPath_rr, PricePath_rr, ParamPath_rr, Parameters, T, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, transpathoptions, simoptions);
    AggVarsPath.(revealperiodnames{rr})=AggVarsPath_rr;

    for ii=1:N_i
        if rr==1
            for aa=1:length(AggVarNames)
                RealizedAggVarsPath.(AggVarNames{aa}).Mean=zeros(1,historylength);
            end
        end

        for aa=1:length(AggVarNames)
            temp=RealizedAggVarsPath.(AggVarNames{aa}).Mean;
            temp2=AggVarsPath_rr.(AggVarNames{aa}).Mean;
            if rr<nReveals
                temp(revealperiods(rr):revealperiods(rr+1)-1)=temp2(1:durationofreveal_rr);
            else
                temp(revealperiods(rr):end)=temp2(1:durationofreveal_rr);
            end
            RealizedAggVarsPath.(AggVarNames{aa}).Mean=temp;
        end
    end
end


end