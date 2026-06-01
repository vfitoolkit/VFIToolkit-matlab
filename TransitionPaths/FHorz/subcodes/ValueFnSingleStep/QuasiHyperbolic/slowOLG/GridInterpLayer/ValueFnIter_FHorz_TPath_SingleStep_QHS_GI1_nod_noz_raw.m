function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar. Policy is QH choice; V output is Vunderbar.
% Vhat is the agent's-perspective value (beta0*beta) at the QH policy.

N_a=prod(n_a);

Policy=zeros(2,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_j,'gpuArray');
Vhat=zeros(N_a,N_j,'gpuArray');

%%
if vfoptions.lowmemory>0
    error('vfoptions.lowmemory>0 not supported for ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_nod_noz_raw')
end

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

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

Vhat(:,N_j)=V(:,N_j);


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    % --- Vhat search (beta0*beta) ---
    entireRHS=ReturnMatrix+beta0beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

    isInfLower    = (ReturnMatrix_L2(1,    :) == -Inf);
    isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    PolicyL2flag(1,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    Policy(1,:,jj)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);

    linidx=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vhat(:,jj)=shiftdim(Vtempii,1);
    V(:,jj)=shiftdim(Vtempii,1)+(beta-beta0beta)*EV_at_policy;
end


%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
