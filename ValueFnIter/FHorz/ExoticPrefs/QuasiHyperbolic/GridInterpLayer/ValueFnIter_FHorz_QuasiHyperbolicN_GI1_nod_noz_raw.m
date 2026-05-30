function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI1_nod_noz_raw.
% No d variables. No z variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(2,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    [~,maxindex]=max(ReturnMatrix,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

    isInfLower    = (ReturnMatrix_ii(1,    :) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);

    Vtilde=V;

else
    % Using V_Jplus1 (V for naive)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    %% V (beta)
    entireRHS=ReturnMatrix+beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    %% Vtilde (beta0*beta)
    entireRHS=ReturnMatrix+beta0beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

    isInfLower    = (ReturnMatrix_L2(1,    :) == -Inf);
    isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Vtilde(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVsource=V(:,jj+1);
    EV=EVsource;

    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    %% V (beta)
    entireRHS=ReturnMatrix+beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    %% Vtilde (beta0*beta)
    entireRHS=ReturnMatrix+beta0beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

    isInfLower    = (ReturnMatrix_L2(1,    :) == -Inf);
    isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Vtilde(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
