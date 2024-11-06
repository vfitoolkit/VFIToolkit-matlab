function AggVarsPath=TransitionPath_FHorz_PType_singlepath(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist,jequalOneDist_T,AgeWeights_T,l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions)
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
            for ttr=1:T-1 % so tt=T-ttr
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyIndexesPath(:,:,T-ttr)=Policy;
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

                AggVarsPath(:,tt)=AggVars;

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

                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end
            
        else % N_e>0
            error('e without z not yet implemented for TPath with FHorz')
        end
    else % N_z>0
        if N_e==0 % z, no e, fastOLG=0
            %% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            PolicyIndexesPath=zeros(N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            % Note that we don't need to keep V for anything
            V=V_final;
            for ttr=1:T-1 % so tt=T-ttr
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in vfoptions already


                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_grid, a_grid,z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyIndexesPath(:,:,:,T-ttr)=Policy;
            end
            % Free up space on GPU by deleting things no longer needed
            clear V
            

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path
            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,:,tt);

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

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
                    pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==0 % Does not depend on tt

                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,N_j,daprime_gridvals,a_gridvals,z_gridvals_J,0,0);

                AggVarsPath(:,tt)=AggVars;

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,:,1:end-1),[1,N_a*N_z*(N_j-1)])),N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,:,1:end-1)/N_d),[1,N_a*N_z*(N_j-1)])),N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                end

                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end
        else  % z, e, fastOLG=0
            %% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            PolicyIndexesPath=zeros(N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            % Note that we don't need to keep V for anything
            V=V_final;
            for ttr=1:T-1 % so tt=T-ttr
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                end
                if transpathoptions.epathtrivial==0
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
            end
            % Free up space on GPU by deleting things no longer needed
            clear V
            

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path
            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,:,:,tt);

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

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
                    pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on tt
                if transpathoptions.epathtrivial==0
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,tt);
                    pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
                end
                % transpathoptions.epathtrivial==1 % Does not depend on tt

                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLGe(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,n_e,N_j,daprime_gridvals,a_gridvals,permute(z_gridvals_J,[3,1,2]),permute(e_gridvals_J,[3,4,1,2]),0,1);

                AggVarsPath(:,tt)=AggVars;

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);

                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(permute(Policy(:,:,:,1:end-1),[1,4,2,3]),[1,N_a*(N_j-1)*N_z*N_e])),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(permute(ceil(Policy(:,:,:,1:end-1)/N_d),[1,4,2,3]),[1,N_a*(N_j-1)*N_z*N_e])),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                end

                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end
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

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path

            % AgentDistPath=zeros([N_a*N_j,T]); 

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

                % AgentDistPath(:,tt)=AgentDist;

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,1:end-1),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,1:end-1)/N_d),[1,N_a*(N_j-1)])),N_a,N_j,jequalOneDist);
                end

                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end

        else % N_e>0
            error('e without z not yet implemented for TPath with FHorz')
        end
    else % N_z>0
        if N_e==0 % z, no e, fastOLG=1
            PolicyIndexesPath=zeros(N_a,N_j,N_z,T-1,'gpuArray'); %Periods 1 to T-1

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

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J(:,:,T-ttr);
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z,z')
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on tt

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [N_a,N_j,N_z] and contains the joint-index for (d,aprime)

                PolicyIndexesPath(:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z
            end
            % Free up space on GPU by deleting things no longer needed
            clear V

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path

            % AgentDistPath=zeros([N_a*N_j*N_z,T]);

            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,:,tt);

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
                
                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
                    pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on tt

                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,N_j,daprime_gridvals,a_gridvals,z_gridvals_J,1,1);

                AggVarsPath(:,tt)=AggVars;

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);

                % AgentDistPath(:,tt)=AgentDist;

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,1:end-1,:),[1,N_a*(N_j-1)*N_z])),N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % Policy for jj=1:N_j-1
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,1:end-1,:)/N_d),[1,N_a*(N_j-1)*N_z])),N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % Policy for jj=1:N_j-1
                end

                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end


        else % N_e>0
            %% z, e, fastOLG=1
             PolicyIndexesPath=zeros(N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1

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

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J(:,:,T-ttr);
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z,z')
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on tt
                if transpathoptions.epathtrivial==0
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % (a,j,z)-by-e
                end
                % transpathoptions.epathtrivial==1 % Does not depend on tt

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [N_a,N_j,N_z,N_e] and contains the joint-index for (d,aprime)

                PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e
            end
            % Free up space on GPU by deleting things no longer needed
            clear V

            %% Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path

            % AgentDistPath=zeros([N_a*N_j*N_z,N_e,T]);

            % Call AgentDist the current periods distn
            for tt=1:T-1

                %Get the current optimal policy
                Policy=PolicyIndexesPath(:,:,:,tt);

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

                if transpathoptions.zpathtrivial==0
                    z_gridvals_J=transpathoptions.z_gridvals_J(:,:,tt);
                    pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on tt
                if transpathoptions.epathtrivial==0
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,tt);
                    pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % (a,j,z)-by-e
                end
                % transpathoptions.epathtrivial==1 % Does not depend on tt

                AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLGe(AgentDist,Policy, FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,l_d,n_a,n_z,n_e,N_j,daprime_gridvals,a_gridvals,z_gridvals_J,e_gridvals_J,1,1);

                AggVarsPath(:,tt)=AggVars;

                % if transpathoptions.ageweightstrivial==0 is hardcoded
                AgeWeightsOld=AgeWeights;
                AgeWeights=AgeWeights_T(:,tt);
                % if transpathoptions.trivialjequalonedist==0 is hardcoded
                jequalOneDist=jequalOneDist_T(:,tt);

                % AgentDistPath(:,:,tt)=AgentDist;

                % if simoptions.fastOLG=1 is hardcoded
                if N_d==0
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(Policy(:,1:end-1,:),[1,N_a*(N_j-1)*N_z*N_e])),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                else
                    % Note, difference is that we do ceil(Policy/N_d) so as to just pass optaprime
                    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,gather(reshape(ceil(Policy(:,1:end-1,:)/N_d),[1,N_a*(N_j-1)*N_z*N_e])),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                end
            
                % Sometimes, we need AggVars to be kept as structure in Parameters (when doing PType_singlepath AggVars is a vector, whereas the rest of the time it is a structure, hence this bit is only needed in PType_singlepath)
                if use_tminus1AggVars==1
                    for aa=1:length(AggVarNames)
                        Parameters.(AggVarNames{aa})=AggVars(aa);
                    end
                end

            end

        end
    end
end
