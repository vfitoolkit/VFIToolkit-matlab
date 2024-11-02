function AggVarsPath=TransitionPath_FHorz_PType_singlepath(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist,jequalOneDist_T,AgeWeights_T,l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, vfoptions, simoptions)
% When doing shooting alogrithm on TPath FHorz PType, this is for a given
% ptype, and does the steps of back-iterate to get policy, then forward to
% get agent dist and agg vars.
% The only output is the agg vars path.

% Note: the input AgentDist, is AgentDist_init

% Note: use AgeWeights_T as transpathoptions.ageweightstrivial==0 is hardcoded
AgeWeights=AgeWeights_T(:,1); % AgeWeights_T is (a,j)-by-T

AggVarsPath=zeros(length(FnsToEvaluate),T-1);


if transpathoptions.fastOLG==0
    if N_z==0
        if N_e==0
            %% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            PolicyIndexesPath=zeros(N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            % Note that we don't need to keep V for anything
            V=V_final;
            for tt=1:T-1 % so t=T-i
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyIndexesPath(:,:,T-tt)=Policy;
            end
            % Free up space on GPU by deleting things no longer needed
            clear V
            

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path
            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,tt);

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if use_tminus1price==1
                    for pp=1:length(tminus1priceNames)
                        if tt>1
                            Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                        else
                            Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                        end
                    end
                end
                if use_tminus1params==1
                    for pp=1:length(tminus1paramNames)
                        if tt>1
                            Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                        else
                            Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                        end
                    end
                end
                if use_tplus1price==1
                    for pp=1:length(tplus1priceNames)
                        kk=tplus1pricePathkk(pp);
                        Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                    end
                end
                if use_tminus1AggVars==1
                    for pp=1:length(tminus1AggVarsNames)
                        if tt>1
                            % The AggVars have not yet been updated, so they still contain previous period values
                            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                        else
                            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                        end
                    end
                end

                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,N_j,daprime_gridvals,a_gridvals,0);

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,1:end-1),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,1:end-1)/N_d),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                end

                AggVarsPath(:,tt)=AggVars;
            end
            
        else % N_e>0
            error('e without z not yet implemented for TPath with FHorz')
        end
    else % N_z>0
        if N_e==0

        else % N_e>0

        end
    end

elseif transpathoptions.fastOLG==1
    if N_z==0
        if N_e==0
            PolicyIndexesPath=zeros(N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1

            %% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step. 
            % Since we won't need to keep the value functions for anything later we just overwrite V
            V=V_final;
            for ttr=1:T-1 % so tt=T-ttr
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end
                
                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The V input is next period value fn, the V output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyIndexesPath(:,:,T-ttr)=Policy;
            end
            % Free up space on GPU by deleting things no longer needed
            clear V
            
            save Vtest.mat V_final % DEBUG
            save Policy.mat PolicyIndexesPath % DEBUG

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path

            AgentDistPath=zeros([N_a*N_j,T]); % Just for debug

            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,tt);

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end
                if use_tminus1price==1
                    for pp=1:length(tminus1priceNames)
                        if tt>1
                            Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                        else
                            Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                        end
                    end
                end
                if use_tminus1params==1
                    for pp=1:length(tminus1paramNames)
                        if tt>1
                            Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                        else
                            Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                        end
                    end
                end
                if use_tplus1price==1
                    for pp=1:length(tplus1priceNames)
                        kk=tplus1pricePathkk(pp);
                        Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                    end
                end
                if use_tminus1AggVars==1
                    for pp=1:length(tminus1AggVarsNames)
                        if tt>1
                            % The AggVars have not yet been updated, so they still contain previous period values
                            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                        else
                            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                        end
                    end
                end
                
                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(gpuArray(AgentDist),Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,N_j,daprime_gridvals,a_gridvals,0);

                AggVarsPath(:,tt)=AggVars;

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);


                AgentDistPath(:,tt)=AgentDist; % JUST FOR DEBUG. DELETE WHEN DONE

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,1:end-1),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,1:end-1)/N_d),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                end
            end


            save AgentDist.mat AgentDistPath
            save AggVarsPath.mat AggVarsPath
            save AgeWeightsPath.mat AgeWeights_T

        else % N_e>0
            error('e without z not yet implemented for TPath with FHorz')
        end
    else % N_z>0
        if N_e==0

        else % N_e>0

        end
    end
end
