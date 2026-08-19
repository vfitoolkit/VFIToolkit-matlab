function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_GI2A_nod1_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_e, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% SemiExo graft of ValueFnIter_FHorz_ExpAssetz_GI2A_nod1_e_raw (which supplies the GI2A grid-interp math).
% d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is the grid-interpolated standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% aprimeFn = aprimeFn(d2, a3, z, ...)   (depends on the current Markov z only, never semiz nor e)
% Policy stores (d2, d3, a1prime-midpoint, a2prime, a1prime-L2index) plus the appended L2flag row.
% lowmemory: 3 shocks {z,semiz,e} => levels {0,1,2,3}.
%   =0 vectorise bothz and e; =1 loop e (bothz parallel); =2 outer-loop z / inner-loop e (semiz parallel); =3 joint bothz outer / inner-loop e.
% EV convention follows this family's SemiExo GI1 raws: expectation over bothzprime first, then the
% 2-corner interp onto a3prime (aprimeFn sees the current z only, so the aprime indices are d3-independent).
% As aprimeFn does not depend on e, DiscountedEV has no e dimension and broadcasts over e.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(5,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d2, d3, a1prime-midpoint, a2prime, a1primeL2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];


if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=2 split over z)
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate (for the d3-loop sections, which loop over d3 and then max over d3)
V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy4_ford3_hat=zeros(4,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d2, a1prime-midpoint, a2prime, a1primeL2ind)
flag_ford3_hat=2*ones(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);
            midpoint_hat=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint_hat(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                midpoint_hat=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint_hat(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3e,[],2);
                    midpoint_hat=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3e,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=midpoint_hat(allind);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex]=max(ReturnMatrix_d3ze,[],2);
                    midpoint_hat=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3ze,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint_hat(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end
    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint_hat
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower grid point index and its weight)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % expand the current z dependence to bothz (semiz fastest)
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs,1,1,N_semiz);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat; % broadcasts over e
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_hat(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]); % broadcasts over e
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_under(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_bothz)-1),-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(N_bothz)*(0:1:(N_e)-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_hat(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_under(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_bothz)-1),-1);
            V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock); % copy, as the skipinterp zeros differ by (d3,z)
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z); % [N_d2*N_a1*N_a2,N_a3,N_semiz]

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

                DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_hat(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_under(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z_hat;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_z_hat(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_z_under(aprime),[N_d2*n2long*N_a2,N_a]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1);
            V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
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
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3)); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_z] (lower grid point index and its weight)

    a1_col=repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col=repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_z]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % expand the current z dependence to bothz (semiz fastest)
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1*N_a2,N_a3,N_bothz]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs,1,1,N_semiz);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat; % broadcasts over e
            [~,maxindex]=max(entireRHS_d3,[],2);
            midpoint=max(min(maxindex,N_a1-1),2);

            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_hat(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]); % broadcasts over e
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_under(aprimez),[N_d2*n2long*N_a2,N_a,N_bothz,N_e]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_bothz)-1),-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(N_bothz)*(0:1:(N_e)-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d2)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

            allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind + N_d2*N_a2*N_a*N_bothz*eBind;
            Policy4_ford3_hat(1,:,:,:,d3_c)=d_ind;
            Policy4_ford3_hat(2,:,:,:,d3_c)=midpoint(allind);
            Policy4_ford3_hat(3,:,:,:,d3_c)=maxindexL2a2;
            Policy4_ford3_hat(4,:,:,:,d3_c)=maxindexL2a1;

            linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind + N_d2*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower   =(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper   =(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            flag_ford3_hat(:,:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat;
                [~,maxindex]=max(entireRHS_d3e,[],2);
                midpoint=max(min(maxindex,N_a1-1),2);

                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_hat(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_under(aprime),[N_d2*n2long*N_a2,N_a,N_bothz]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_bothz)-1),-1);
            V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtempii,1);

                d_ind        =rem(maxindexL2-1,N_d2)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*bothzBind;
                Policy4_ford3_hat(1,:,:,e_c,d3_c)=d_ind;
                Policy4_ford3_hat(2,:,:,e_c,d3_c)=midpoint(allind);
                Policy4_ford3_hat(3,:,:,e_c,d3_c)=maxindexL2a2;
                Policy4_ford3_hat(4,:,:,e_c,d3_c)=maxindexL2a1;

                linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*bothzBind;
                isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                flag_ford3_hat(:,:,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock); % copy, as the skipinterp zeros differ by (d3,z)
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z); % [N_d2*N_a1*N_a2,N_a3,N_semiz]

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]);
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

                DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV_hat;
                    [~,maxindex]=max(entireRHS_d3e,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_semiz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3e=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_hat(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3e,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=reshape(ReturnMatrix_ii_d3e+DiscountedEVinterp_under(aprime),[N_d2*n2long*N_a2,N_a,N_semiz]);
                maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1)+shiftdim((N_d2*n2long*N_a2)*(N_a)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind + N_d2*N_a2*N_a*semizBind;
                    Policy4_ford3_hat(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,semizblock,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,semizblock,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,semizblock,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind + N_d2*n2long*N_a2*N_a*semizBind;
                    isInfLower   =(ReturnMatrix_ii_d3e(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3e(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end

    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full; % copy, as the skipinterp zeros differ by d3
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2*N_a1*N_a2,N_a3,N_bothz]

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_grid,permute(DiscountedEV_under,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            DiscountedEVinterp_hat=permute(interp1(a1_grid,permute(DiscountedEV_hat,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z_hat;
                    [~,maxindex]=max(entireRHS_d3ze,[],2);
                    midpoint=max(min(maxindex,N_a1-1),2);

                    a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, [n_d2,1], n_a2, n_a3, special_n_bothz, special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexes), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                    aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_d3ze=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_z_hat(aprime),[N_d2*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii_d3ze,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=reshape(ReturnMatrix_ii_d3ze+DiscountedEVinterp_z_under(aprime),[N_d2*n2long*N_a2,N_a]);
            maxindexfull=maxindexL2+(N_d2*n2long*N_a2)*(0:1:(N_a)-1);
            V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);

                    d_ind        =rem(maxindexL2-1,N_d2)+1;
                    maxindexL2a1 =rem(floor((maxindexL2-1)/N_d2),n2long)+1;
                    maxindexL2a2 =floor((maxindexL2-1)/(N_d2*n2long))+1;

                    allind=d_ind + N_d2*(maxindexL2a2-1) + N_d2*N_a2*aind;
                    Policy4_ford3_hat(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy4_ford3_hat(2,:,z_c,e_c,d3_c)=midpoint(allind);
                    Policy4_ford3_hat(3,:,z_c,e_c,d3_c)=maxindexL2a2;
                    Policy4_ford3_hat(4,:,z_c,e_c,d3_c)=maxindexL2a1;

                    linidx_lower=d_ind                   + N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    linidx_upper=d_ind + N_d2*(n2long-1)+ N_d2*n2long*(maxindexL2a2-1) + N_d2*n2long*N_a2*aind;
                    isInfLower   =(ReturnMatrix_ii_d3ze(linidx_lower)==-Inf);
                    isInfUpper   =(ReturnMatrix_ii_d3ze(linidx_upper)==-Inf);
                    inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                    inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                    flag_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                end
            end
        end
    end

    % Max over d3 and unpack
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=4*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz,N_e]); % d2
    Policy(3,:,:,:,jj)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz,N_e]); % a1prime midpoint
    Policy(4,:,:,:,jj)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz,N_e]); % a2prime
    Policy(5,:,:,:,jj)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz,N_e]); % a1primeL2ind
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(5,:,:,:,:)=adjust.*Policy(5,:,:,:,:)+(1-adjust).*(Policy(5,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
