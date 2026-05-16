function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_noz_raw.
% No d variables. No z variable. No e variable. GPU (parallel==2 only).
%
% Naive: V_j = max u + beta*E[V_{j+1}]  (used as EVsource)
%        Vtilde_j = max u + beta_0*beta*E[V_{j+1}]  (agent's choice)

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Vtilde=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray');
Policy_V=zeros(N_a,N_j,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(Policy(level1ii(ii),N_j):Policy(level1ii(ii+1),N_j)), a_grid(curraindex), ReturnFnParamsVec);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+Policy(level1ii(ii),N_j)-1;
    end
    Vtilde=V;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    V(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy_V(level1ii,N_j)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy_V(level1ii(ii),N_j):Policy_V(level1ii(ii+1),N_j);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy_V(curraindex,N_j)=shiftdim(maxindex,1)+Policy_V(level1ii(ii),N_j)-1;
    end
    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    Vtilde(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy(level1ii(ii),N_j):Policy(level1ii(ii+1),N_j);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        Vtilde(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+Policy(level1ii(ii),N_j)-1;
    end
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

    EV=V(:,jj+1);

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    %% V (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy_V(level1ii,jj)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy_V(level1ii(ii),jj):Policy_V(level1ii(ii+1),jj);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,jj)=shiftdim(Vtempii,1);
        Policy_V(curraindex,jj)=shiftdim(maxindex,1)+Policy_V(level1ii(ii),jj)-1;
    end
    %% Vtilde (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    Vtilde(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy(level1ii(ii),jj):Policy(level1ii(ii+1),jj);
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        Vtilde(curraindex,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,jj)=shiftdim(maxindex,1)+Policy(level1ii(ii),jj)-1;
    end
end

end
