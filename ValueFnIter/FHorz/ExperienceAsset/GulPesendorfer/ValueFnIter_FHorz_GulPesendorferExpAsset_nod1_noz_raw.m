function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noz_raw(n_d2,n_a1,n_a2,N_j, d2_gridvals, a1_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions)
% Gul-Pesendorfer with an experience asset: V_j = max_{d2,a1'} [u+v+beta*E V_{j+1}] - max_{d2,a1'} v.
% The tempted objective goes through the standard ExpAsset machinery (the continuation is at
% a2prime=aprimeFn(d2,a2), interpolated onto the a2 grid); the most-tempting term is a max of v
% over the FULL joint (d2,a1prime) choice set and is subtracted AFTER the max (v never touches
% the continuation, so the temptation side needs none of the aprime-probs machinery).

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

% n_a1prime=n_a;
% a1prime_gridvals=a1_gridvals;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,0,0); % Level=0, Refine=0
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    entireRHS=ReturnMatrix+TemptationMatrix;
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp-MostTempting;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    % l_a2==1: [N_d2,N_a2] legacy; l_a2>1: [l_a2,N_d2,N_a2] per-dim factored (NOT a Kron fold)

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1);
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1);

        Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2]);
        Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % a2primeIndex shape [l_a2=2, N_d2, N_a2]: per-dim lower-grid idx
        % a2primeProbs same shape: per-dim prob of lower
        % Do nested 2-corner interp with skipinterp at each level (bit-exact when V is flat in a dim)
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[N_d2,N_a2]),N_a1,1);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[N_d2,N_a2]),N_a1,1);

        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2); % [N_d2*N_a1, N_a2]
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
        V_ll=reshape(EVpre(aprime_ll(:)),[N_d2*N_a1,N_a2]);
        V_hl=reshape(EVpre(aprime_hl(:)),[N_d2*N_a1,N_a2]);
        V_lh=reshape(EVpre(aprime_lh(:)),[N_d2*N_a1,N_a2]);
        V_hh=reshape(EVpre(aprime_hh(:)),[N_d2*N_a1,N_a2]);

        % Step 1a: interp in a2_1 at a2_2=lo with skipinterp; per-contribution NaN cleanup
        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        % Step 1b: interp in a2_1 at a2_2=hi
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        % Step 2: interp in a2_2
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV(isnan(EV))=0;

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0,n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0,n_d2, n_a1, n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,0,0); % Level=0, Refine=0
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp-MostTempting,1);
    Policy(:,N_j)=shiftdim(maxindex,1);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    % l_a2==1: [N_d2,N_a2] legacy; l_a2>1: [l_a2,N_d2,N_a2] per-dim factored (NOT a Kron fold)

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1);
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1);

        Vlower=reshape(V(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2]);
        Vupper=reshape(V(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % a2primeIndex/a2primeProbs shape [l_a2=2, N_d2, N_a2] per-dim. Nested 2-corner with skipinterp.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[N_d2,N_a2]),N_a1,1);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[N_d2,N_a2]),N_a1,1);

        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
        V_ll=reshape(V(aprime_ll(:),jj+1),[N_d2*N_a1,N_a2]);
        V_hl=reshape(V(aprime_hl(:),jj+1),[N_d2*N_a1,N_a2]);
        V_lh=reshape(V(aprime_lh(:),jj+1),[N_d2*N_a1,N_a2]);
        V_hh=reshape(V(aprime_hh(:),jj+1),[N_d2*N_a1,N_a2]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV(isnan(EV))=0;

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, 0, n_d2, n_a1,n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, 0, n_d2, n_a1,n_a1,n_a2, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,0,0); % Level=0, Refine=0
    MostTempting=max(TemptationMatrix,[],1);
    MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN

    entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp-MostTempting,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%%
Policy=shiftdim(Policy,-1);



end
