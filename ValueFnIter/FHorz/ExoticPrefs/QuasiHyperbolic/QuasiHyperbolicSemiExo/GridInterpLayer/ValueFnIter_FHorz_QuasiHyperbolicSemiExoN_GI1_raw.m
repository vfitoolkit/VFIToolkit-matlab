function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic + SemiExo + GridInterpLayer raw (with d1, with z).
% Splice of ValueFnIter_FHorz_QuasiHyperbolicN_GI1_raw and ValueFnIter_FHorz_SemiExo_GI1_raw.
% Per-d2 EV with pi_bothz=kron(pi_z, pi_semiz(d2)); midpoint+L2 inside d2 loop for both V and Vtilde.
% L2flag detection runs on Vtilde's (beta0beta) Return matrix.

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

V=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz*N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz*N_z,N_j,'gpuArray'); % [d1; d2; midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_semiz*N_z,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

aind=gpuArray(0:1:N_a-1);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-2);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate per-d2 slabs
V_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');           % V (beta)
Valt_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');        % Vtilde (beta0beta)
Policy_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');      % d1*aprimeL2ind packed
midpoint_ford2_jj=zeros(N_a,N_semiz*N_z,N_d2,'gpuArray');
flag_ford2_jj=2*ones(N_a,N_semiz*N_z,N_d2,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period: just standard Return-max (single max, used for both V and Vtilde).
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_bothz, d_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_bothz,d_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind;
        L2offset      = ceil(maxindexL2/N_d);
        linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
        linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        Policy(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_bothz, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_bothz,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            L2offset      = ceil(maxindexL2/N_d);
            linidx_lower  = d_ind                  + N_d*n2long*aind;
            linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            Policy(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(4,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
    Vtilde(:,:,N_j)=V(:,:,N_j);
else
    % Using V_Jplus1 (V for naive)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_bothz, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            %% V (beta) -- per-d2 V slab
            entireRHS=ReturnMatrix_d2+beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpointV=max(min(maxindex,n_a-1),2);
            aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimezV=aprimeindexesV+n2aprime*bothzBind;
            entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,~]=max(entireRHS_iiV,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);

            %% Vtilde (beta0beta) -- per-d2 Vtilde slab + policy + L2flag
            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));

            % L2 flag (per d2): detect -Inf on Vtilde's Return matrix
            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_d2z=EVpre.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_bothz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

                %% V (beta)
                entireRHS_z=ReturnMatrix_d2z+beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpointV=max(min(maxindex,n_a-1),2);
                aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp_z(aprimeindexesV),[N_d1*n2long,N_a]);
                [Vtemp,~]=max(entireRHS_iiV,[],1);
                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);

                %% Vtilde (beta0beta)
                entireRHS_z=ReturnMatrix_d2z+beta0beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Valt_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind;
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
    end

    %% Cross-d2 gather: V picks its own d2 winner; Vtilde picks its own d2 winner and drives Policy.
    [V_jj,~]=max(V_ford2_jj,[],3);
    V(:,:,N_j)=V_jj;
    [Vtilde_jj,maxindex]=max(Valt_ford2_jj,[],3);
    Vtilde(:,:,N_j)=Vtilde_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d2
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1);
    Policy(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1);
    Policy(3,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
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

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_bothz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_bothz, d12c_gridvals, a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            %% V (beta)
            entireRHS=ReturnMatrix_d2+beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpointV=max(min(maxindex,n_a-1),2);
            aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimezV=aprimeindexesV+n2aprime*bothzBind;
            entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp(aprimezV),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,~]=max(entireRHS_iiV,[],1);
            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);

            %% Vtilde (beta0beta)
            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzBind;
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Valt_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind;
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));

            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_d2z=EVpre.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_bothz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

                %% V (beta)
                entireRHS_z=ReturnMatrix_d2z+beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpointV=max(min(maxindex,n_a-1),2);
                aprimeindexesV=(midpointV+(midpointV-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2iiV=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexesV), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_iiV=ReturnMatrix_d2iiV+beta*reshape(EVinterp_z(aprimeindexesV),[N_d1*n2long,N_a]);
                [Vtemp,~]=max(entireRHS_iiV,[],1);
                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);

                %% Vtilde (beta0beta)
                entireRHS_z=ReturnMatrix_d2z+beta0beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Valt_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind;
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            end
        end
    end

    %% Cross-d2 gather
    [V_jj,~]=max(V_ford2_jj,[],3);
    V(:,:,jj)=V_jj;
    [Vtilde_jj,maxindex]=max(Valt_ford2_jj,[],3);
    Vtilde(:,:,jj)=Vtilde_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    Policy(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1);
    Policy(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1);
    Policy(3,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz*N_z)'+(N_a*N_semiz*N_z)*(maxindex-1)),[1,N_a,N_semiz*N_z]);
end

%% Post-process Policy: convert midpoint+aprimeL2ind to lower-grid + offset
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

end
