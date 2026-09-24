function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetS_GI2A_noz_raw(n_d1, n_d2, n_a1, n_a2, n_a3, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_ExpAsset_GI2A_noz_raw.
% a1=standard endogenous state carrying the grid interpolation layer, a2=folded
% standard endogenous state(s), a3=experience asset. GPU only.
% Policy is 4-channel: 1=d, 2=a1prime midpoint, 3=a2prime, 4=a1prime L2;
% the L2 flag is channel 5.
%
% Sophisticated: Vhat_j      = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVfine_at_optimal_choice
% EVfine is the (undiscounted) interpolated continuation actually added to the layer-2
% RHS, so the a3 lottery is already baked in and the gather needs no lottery handling.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;

Vhat=zeros(N_a,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_j,'gpuArray'); % exponential value at the QH policy
Policy=zeros(5,N_a,N_j,'gpuArray'); % 1=d (joint), 2=a1prime midpoint, 3=a2prime, 4=a1prime L2 fine
Policy(5,:,:)=2;

%% GI setup
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d, 1]
aind=gpuArray(0:1:N_a-1);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    [~,maxindex]=max(ReturnMatrix,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Vunderbar(:,N_j)=Vhat(:,N_j); % terminal: no continuation, so Vunderbar equals Vhat

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    % --- Vhat search (beta0*beta) ---
    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    EVfine=reshape(entireEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    entireRHS_ii=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a])+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=midpoint(allind);
    Policy(3,:,N_j)=maxindexL2a2;
    Policy(4,:,N_j)=maxindexL2a1;

    linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
    linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,N_j)=Vhat(:,N_j)+(beta-beta0beta)*EV_at_policy;
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a3, d2_gridvals, a3_grid, aprimeFnParamsVec,2);

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1);

    Vlower=reshape(Vunderbar(aprimeIndex(:),jj+1),    [N_d2*N_a1*N_a2,N_a3]);
    Vupper=reshape(Vunderbar(aprimeplus1Index(:),jj+1),[N_d2*N_a1*N_a2,N_a3]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);

    entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3]); % undiscounted; beta/beta0beta applied at use sites
    entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 1);
    % --- Vhat search (beta0*beta) ---
    entireRHS=ReturnMatrix+beta0beta*repelem(entireEV,N_d1,1,1,1,1,1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,N_a1-1),2);

    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, ReturnFnParamsVec, 3);
    aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
    EVfine=reshape(entireEVinterp(aprime),[N_d*n2long*N_a2,N_a]);
    entireRHS_ii=reshape(ReturnMatrix_ii,[N_d*n2long*N_a2,N_a])+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
    Vhat(:,jj)=shiftdim(Vtempii,1);

    d_ind        =rem(maxindexL2-1,N_d)+1;
    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
    maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

    allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=midpoint(allind);
    Policy(3,:,jj)=maxindexL2a2;
    Policy(4,:,jj)=maxindexL2a1;

    linidx_lower=d_ind                + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    linidx_upper=d_ind + N_d*(n2long-1)+ N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
    isInfLower   =(ReturnMatrix_ii(linidx_lower)==-Inf);
    isInfUpper   =(ReturnMatrix_ii(linidx_upper)==-Inf);
    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
    Policy(5,:,jj)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    % --- Vunderbar: exponential value at the QH-chosen (interpolated) point ---
    linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,jj)=Vhat(:,jj)+(beta-beta0beta)*EV_at_policy;
end


%% Post-process
adjust=(Policy(4,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(4,:,:)=Policy(4,:,:)-(n2short+1)*(~adjust);

end
