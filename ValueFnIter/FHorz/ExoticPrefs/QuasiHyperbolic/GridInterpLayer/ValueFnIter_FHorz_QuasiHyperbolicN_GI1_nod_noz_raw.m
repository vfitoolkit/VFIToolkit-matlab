function varargout=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI_nod_noz_raw.
% No d variables. No z variable. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}]   (agent's choice)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(2,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind]

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    [~,maxindex]=max(ReturnMatrix,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
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

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    %% V (beta)
    entireRHS=ReturnMatrix+beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2,[],1);
    V(:,N_j)=shiftdim(Vtempii,1);
    %% Vtilde (beta0*beta)
    entireRHS=ReturnMatrix+beta0beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
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

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    %% V (beta)
    entireRHS=ReturnMatrix+beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,~]=max(entireRHS_L2,[],1);
    V(:,jj)=shiftdim(Vtempii,1);
    %% Vtilde (beta0*beta)
    entireRHS=ReturnMatrix+beta0beta*EV;
    [~,maxindex]=max(entireRHS,[],1);
    midpoint=max(min(maxindex,n_a-1),2);
    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vtilde(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoint),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);
end

%% Post-process Policy: convert [midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:)+N_a*(Policy(2,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vtilde,Policy};
elseif nOutputs==3
    varargout={Vtilde,Policy,V};
end

end
