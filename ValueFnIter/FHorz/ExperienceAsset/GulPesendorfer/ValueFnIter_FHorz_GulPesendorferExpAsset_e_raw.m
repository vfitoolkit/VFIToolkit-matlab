function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions)
% Gul-Pesendorfer with an experience asset: V_j = max_{d1,d2,a1'} [u+v+beta*E V_{j+1}] - max_{d1,d2,a1'} v.
% The tempted objective goes through the standard ExpAsset machinery (the continuation is at
% a2prime=aprimeFn(d2,a2), interpolated onto the a2 grid); the most-tempting term is a max of v
% over the FULL joint (d1,d2,a1prime) choice set and is subtracted AFTER the max (v never touches
% the continuation, so the temptation side needs none of the aprime-probs machinery).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,0,0); % Level=0, Refine=0
        MostTempting=max(TemptationMatrix,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        entireRHS=ReturnMatrix+TemptationMatrix;
        %Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,N_j)=Vtemp-MostTempting;
        Policy(:,:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            TemptationMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting_e=max(TemptationMatrix_e,[],1);
            MostTempting_e(MostTempting_e==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e;
            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,N_j)=Vtemp-MostTempting_e;
            Policy(:,:,e_c,N_j)=maxindex;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                TemptationMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                MostTempting_ze(MostTempting_ze==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze;
                % Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS_ze);
                V(:,z_c,e_c,N_j)=Vtemp-MostTempting_ze;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), TemptationFnParamsVec,0,0); % Level=0, Refine=0
        MostTempting=max(TemptationMatrix,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

        entireRHS=ReturnMatrix+TemptationMatrix+DiscountedEV; % should autofill e dimension

        %Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp-MostTempting,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            TemptationMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting_e=max(TemptationMatrix_e,[],1);
            MostTempting_e(MostTempting_e==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

            entireRHS=ReturnMatrix_e+TemptationMatrix_e+DiscountedEV;

            % Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp-MostTempting_e,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountFactorParamsVec*repelem(EV(:,:,z_c),N_d1,N_a1);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                TemptationMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                MostTempting_ze(MostTempting_ze==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

                entireRHS=ReturnMatrix_ze+TemptationMatrix_ze+DiscountedEV_z;

                % Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,N_j)=shiftdim(Vtemp-MostTempting_ze,1);
                Policy(:,z_c,e_c,N_j)=shiftdim(maxindex,1);
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    EVpre=sum(shiftdim(pi_e_J(:,jj+1),-2).*V(:,:,:,jj+1),3); % First, switch V_Jplus1 into Kron form

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), TemptationFnParamsVec,0,0); % Level=0, Refine=0
        MostTempting=max(TemptationMatrix,[],1);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

        entireRHS=ReturnMatrix+TemptationMatrix+DiscountedEV; % should autofill e dimension

        % Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp-MostTempting,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            TemptationMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting_e=max(TemptationMatrix_e,[],1);
            MostTempting_e(MostTempting_e==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

            entireRHS=ReturnMatrix_e+TemptationMatrix_e+DiscountedEV;

            % Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp-MostTempting_e,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountFactorParamsVec*repelem(EV(:,:,z_c),N_d1,N_a1);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                TemptationMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, TemptationFnParamsVec,0,0); % Level=0, Refine=0
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                MostTempting_ze(MostTempting_ze==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

                entireRHS=ReturnMatrix_ze+TemptationMatrix_ze+DiscountedEV_z;

                %Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,jj)=shiftdim(Vtemp-MostTempting_ze,1);
                Policy(:,z_c,e_c,jj)=shiftdim(maxindex,1);
            end
        end
    end
end

%%
Policy=shiftdim(Policy,-1);



end
