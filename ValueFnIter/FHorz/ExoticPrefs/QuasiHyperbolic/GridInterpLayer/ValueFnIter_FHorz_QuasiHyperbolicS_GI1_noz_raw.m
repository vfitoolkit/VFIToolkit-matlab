function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI1_noz_raw.
% Has d variables. No z variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j   = max_{d,a'} u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVinterp_at_optimal_aprime

N_d=prod(n_d);
N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray');
Policy=zeros(3,N_a,N_j,'gpuArray'); % [d_ind; midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt

aind=gpuArray(0:1:N_a-1);

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);
    [~,maxindex]=max(ReturnMatrix,[],2);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind + N_d*n2long*aind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);

    Vunderbar=Vhat;

else
    % Using V_Jplus1 (should be Vunderbar for sophisticated)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vunderbar=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);

    % --- Vhat search (beta0*beta) ---
    entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind + N_d*n2long*aind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    allind=d_ind+N_d*aind;
    Policy(1,:,N_j)=d_ind;
    Policy(2,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    linidx=reshape(maxindexL2,[1,N_a])+N_d*n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,N_j)=Vhat(:,N_j)+(beta-beta0beta)*EV_at_policy;
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

    EVsource=Vunderbar(:,jj+1);
    EV=EVsource;

    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,1);

    % --- Vhat search (beta0*beta) ---
    entireRHS=ReturnMatrix+beta0beta*shiftdim(EV,-1);
    [~,maxindex]=max(entireRHS,[],2);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn,n_d,d_gridvals,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec,2);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[N_d*n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat(:,jj)=shiftdim(Vtempii,1);
    d_ind=rem(maxindexL2-1,N_d)+1;
    L2offset = ceil(maxindexL2/N_d);
    linidx_lower = d_ind + N_d*n2long*aind;
    linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
    PolicyL2flag(1,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
    allind=d_ind+N_d*aind;
    Policy(1,:,jj)=d_ind;
    Policy(2,:,jj)=shiftdim(squeeze(midpoint(allind)),-1);
    Policy(3,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1);
    linidx=reshape(maxindexL2,[1,N_a])+N_d*n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,jj)=Vhat(:,jj)+(beta-beta0beta)*EV_at_policy;
end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:)<1+n2short+1);
Policy(2,:,:)=Policy(2,:,:)-adjust;
Policy(3,:,:)=adjust.*Policy(3,:,:)+(1-adjust).*(Policy(3,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
