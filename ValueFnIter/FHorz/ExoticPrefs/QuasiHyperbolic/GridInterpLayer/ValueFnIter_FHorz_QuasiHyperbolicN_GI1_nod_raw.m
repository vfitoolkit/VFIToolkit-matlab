function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_GI1_nod_raw.
% No d variables. GPU (parallel==2 only).
%
% Naive:  V_j    = max_{a'} u + beta*E[V_{j+1}]          (time-consistent)
%         Vtilde_j = max_{a'} u + beta_0*beta*E[V_{j+1}] (agent's actual choice)

N_a=prod(n_a);
N_z=prod(n_z);

Valt=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); % [midpoint; aprimeL2ind]
PolicyL2flag=2*ones(1,N_a,N_z,N_j,'gpuArray'); % 1=all weight to lower coarse pt, 2=usual linear weights, 3=all weight to upper coarse pt
Policyalt=zeros(2,N_a,N_z,N_j,'gpuArray'); % exponential discounter optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_z,N_j,'gpuArray');

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1); % 1-by-1-by-N_z

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No discounting at terminal period.
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        [~,maxindex]=max(ReturnMatrix,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

        isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            [~,maxindex]=max(ReturnMatrix,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);

            isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Valt(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j); % terminal: QH and exp discounter coincide
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);

else
    % Using V_Jplus1 (Valt for naive)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vtilde=zeros(N_a,N_z,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindexalt]=max(entireRHS,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2alt=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimezalt=aprimeindexesalt+n2aprime*zind;
        entireRHS_L2alt=ReturnMatrix_L2alt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2alt,[],1);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);

        isInfLoweralt    = (ReturnMatrix_L2alt(1,     :,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_L2alt(n2long,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        PolicyL2flagalt(1,:,:,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

        Policyalt(1,:,:,N_j)=shiftdim(squeeze(midpointalt),-1);
        Policyalt(2,:,:,N_j)=shiftdim(maxindexL2alt,-1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS=ReturnMatrix+beta*EV_z;
            [~,maxindexalt]=max(entireRHS,[],1);
            midpointalt=max(min(maxindexalt,n_a-1),2);
            aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2alt=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexesalt),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2alt=ReturnMatrix_L2alt+beta*reshape(EVinterp_z(aprimeindexesalt(:)),[n2long,N_a]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2alt,[],1);
            Valt(:,z_c,N_j)=shiftdim(Vtempii,1);

            isInfLoweralt    = (ReturnMatrix_L2alt(1,     :) == -Inf);
            isInfUpperalt    = (ReturnMatrix_L2alt(n2long,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagalt(1,:,z_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

            Policyalt(1,:,z_c,N_j)=shiftdim(squeeze(midpointalt),-1);
            Policyalt(2,:,z_c,N_j)=shiftdim(maxindexL2alt,-1);
            %% Vtilde (beta0*beta)
            entireRHS=ReturnMatrix+beta0beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

            isInfLower    = (ReturnMatrix_L2(1,     :) == -Inf);
            isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Vtilde(:,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
        end
    end
end

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

    EVsource=Valt(:,:,jj+1);
    EV=EVsource.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);   % N_a-by-1-by-N_z

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
        entireRHS=ReturnMatrix+beta*EV;
        [~,maxindexalt]=max(entireRHS,[],1);
        midpointalt=max(min(maxindexalt,n_a-1),2);
        aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2alt=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexesalt),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimezalt=aprimeindexesalt+n2aprime*zind;
        entireRHS_L2alt=ReturnMatrix_L2alt+beta*reshape(EVinterp(aprimezalt(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2alt,[],1);
        Valt(:,:,jj)=shiftdim(Vtempii,1);

        isInfLoweralt    = (ReturnMatrix_L2alt(1,     :,:) == -Inf);
        isInfUpperalt    = (ReturnMatrix_L2alt(n2long,:,:) == -Inf);
        inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
        inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
        PolicyL2flagalt(1,:,:,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

        Policyalt(1,:,:,jj)=shiftdim(squeeze(midpointalt),-1);
        Policyalt(2,:,:,jj)=shiftdim(maxindexL2alt,-1);
        %% Vtilde (beta0*beta)
        entireRHS=ReturnMatrix+beta0beta*EV;
        [~,maxindex]=max(entireRHS,[],1);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoint),-1);
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            %% Valt (beta) -- capture Policyalt (exponential discounter's choice)
            entireRHS=ReturnMatrix+beta*EV_z;
            [~,maxindexalt]=max(entireRHS,[],1);
            midpointalt=max(min(maxindexalt,n_a-1),2);
            aprimeindexesalt=(midpointalt+(midpointalt-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2alt=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexesalt),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2alt=ReturnMatrix_L2alt+beta*reshape(EVinterp_z(aprimeindexesalt(:)),[n2long,N_a]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2alt,[],1);
            Valt(:,z_c,jj)=shiftdim(Vtempii,1);

            isInfLoweralt    = (ReturnMatrix_L2alt(1,     :) == -Inf);
            isInfUpperalt    = (ReturnMatrix_L2alt(n2long,:) == -Inf);
            inLowerStrictalt = (maxindexL2alt >= 2)         & (maxindexL2alt <= n2short+1);
            inUpperStrictalt = (maxindexL2alt >= n2short+3) & (maxindexL2alt <= n2long-1);
            PolicyL2flagalt(1,:,z_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);

            Policyalt(1,:,z_c,jj)=shiftdim(squeeze(midpointalt),-1);
            Policyalt(2,:,z_c,jj)=shiftdim(maxindexL2alt,-1);
            %% Vtilde (beta0*beta)
            entireRHS=ReturnMatrix+beta0beta*EV_z;
            [~,maxindex]=max(entireRHS,[],1);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);

            isInfLower    = (ReturnMatrix_L2(1,     :) == -Inf);
            isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            PolicyL2flag(1,:,z_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            Vtilde(:,z_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,z_c,jj)=shiftdim(squeeze(midpoint),-1);
            Policy(2,:,z_c,jj)=shiftdim(maxindexL2,-1);
        end
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

Policy=[Policy;PolicyL2flag];

adjustalt=(Policyalt(2,:,:,:)<1+n2short+1);
Policyalt(1,:,:,:)=Policyalt(1,:,:,:)-adjustalt;
Policyalt(2,:,:,:)=adjustalt.*Policyalt(2,:,:,:)+(1-adjustalt).*(Policyalt(2,:,:,:)-n2short-1);

Policyalt=[Policyalt;PolicyL2flagalt];

end
