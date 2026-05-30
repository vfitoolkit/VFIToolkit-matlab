function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive QH + SemiExo + GI: with d1, with z, with e.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

V_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Valt_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');
flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_e,N_d2,'gpuArray');

pi_e_J=shiftdim(pi_e_J,-2);

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind;
        L2offset      = ceil(maxindexL2/N_d);
        linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind + N_d*n2long*N_a*N_bothz*eind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        Policy(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, special_n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            [~,maxindex]=max(ReturnMatrix_e,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*bothzind;
            L2offset      = ceil(maxindexL2/N_d);
            linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            Policy(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
    Vtilde(:,:,:,N_j)=V(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]);
    EVpre=sum(EVpre.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            %% V (beta)
            entireRHS=ReturnMatrix_d2+beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpointV=max(min(maxindex,n_a-1),2);
            aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimezV=aprimeindexesV+n2aprime*bothzBind;
            entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,~]=max(entireRHS_iiV,[],1);
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);

            %% Vtilde (beta0beta)
            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Valt_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind;
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);

                %% V (beta)
                entireRHS_e=ReturnMatrix_e+beta*shiftdim(EV_d2,-1);
                [~,maxindex]=max(entireRHS_e,[],2);
                midpointV=max(min(maxindex,n_a-1),2);
                aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                aprimezV=aprimeindexesV+n2aprime*bothzBind;
                entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,~]=max(entireRHS_iiV,[],1);
                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);

                %% Vtilde (beta0beta)
                entireRHS_e=ReturnMatrix_e+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex]=max(entireRHS_e,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*bothzBind;
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Valt_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
    end

    [V_jj,~]=max(V_ford2_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    [Vtilde_jj,maxindex]=max(Valt_ford2_jj,[],4);
    Vtilde(:,:,:,N_j)=Vtilde_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
    Policy(1,:,:,:,N_j)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
    Policy(4,:,:,:,N_j)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
    Policy(3,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
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

    EVpre=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            %% V (beta)
            entireRHS=ReturnMatrix_d2+beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpointV=max(min(maxindex,n_a-1),2);
            aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimezV=aprimeindexesV+n2aprime*bothzBind;
            entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,~]=max(entireRHS_iiV,[],1);
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);

            %% Vtilde (beta0beta)
            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Valt_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind+N_d1*N_a*N_bothz*eind;
            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoint(allind));

            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);

                %% V (beta)
                entireRHS_e=ReturnMatrix_e+beta*shiftdim(EV_d2,-1);
                [~,maxindex]=max(entireRHS_e,[],2);
                midpointV=max(min(maxindex,n_a-1),2);
                aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                aprimezV=aprimeindexesV+n2aprime*bothzBind;
                entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,~]=max(entireRHS_iiV,[],1);
                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);

                %% Vtilde (beta0beta)
                entireRHS_e=ReturnMatrix_e+beta0beta*shiftdim(EV_d2,-1);
                [~,maxindex]=max(entireRHS_e,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, special_n_d, n_bothz, special_n_e, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*bothzBind;
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Valt_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
    end

    [V_jj,~]=max(V_ford2_jj,[],4);
    V(:,:,:,jj)=V_jj;
    [Vtilde_jj,maxindex]=max(Valt_ford2_jj,[],4);
    Vtilde(:,:,:,jj)=Vtilde_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z*N_e]);
    Policy(1,:,:,:,jj)=reshape(rem(d1aprimeL2_ind-1,N_d1)+1,[N_a,N_semiz*N_z,N_e]);
    Policy(4,:,:,:,jj)=reshape(ceil(d1aprimeL2_ind/N_d1),[N_a,N_semiz*N_z,N_e]);
    Policy(3,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
end

%% Post-process Policy
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
