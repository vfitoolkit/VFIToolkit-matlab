function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizS_DC2A_GI2A_noz_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Sophisticated quasi-hyperbolic discounting + ExperienceAssetsemiz, two standard endogenous
% states: divide-and-conquer on a1 (DC2A) plus the grid interpolation layer on a1 (GI2A); the
% remaining standard endogenous state a2 is folded (choice a2prime); a3 is the experience asset.
% Sophisticated: Vhat_j      = max_{d,a1prime,a2prime} F + beta0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta-beta0*beta)*EVfine_at_the_hat_argmax
% There is a single maximisation (hence a single DC bracket, a single GI midpoint and a single
% folded a2prime choice); the beta-discounted continuation is then GATHERED at that same argmax
% out of EVfine, the undiscounted interpolated continuation that was actually added to the
% layer-2 RHS. The a3 lottery is resolved inside EV before the interpolation, so the gather needs
% no lottery handling. The d3 choice is made on the hat values, and Vunderbar is gathered at that
% same d3. beta0 is received as a trailing input, beta0beta=beta0*beta.
% Outputs [Vhat,Policy,Vunderbar]; the backward recursion uses Vunderbar.
% ExperienceAssetsemiz: aprimeFn=aprimeFn(d2,a3,semiz), so the a3prime lottery carries a
% current-semiz dimension and the EV contraction over the shock-prime index is done inside the
% d3 loop (it cannot be hoisted out of it). The QH dual attaches to the continuation that comes
% out of that contraction, exactly where it attaches in the SemiExo sibling.
% NO z, NO e _noz analog of ValueFnIter_FHorz_ExpAssetsemiz_DC2A_GI2A_raw, keeping the divide-and-conquer + grid-interp math on a1.
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered+grid-interp standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% NO z, NO e: the only shock is semiz (so bothz=semiz throughout).
% Policy is 5-channel: 1=d1, 2=d2, 3=d3, 4=joint(a1prime midpoint->lower grid point after post-process, a2prime), 5=a1prime L2; the L2 flag as 6th.
% lowmemory: 1 shock {semiz} => levels {0,1}.
%   =0 vectorise semiz; =1 loop semiz.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_d=N_d1*N_d2*N_d3;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);

Vhat=zeros(N_a,N_semiz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_j,'gpuArray'); % the beta-discounted value gathered at the hat argmax
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime(mid),a2prime),a1primeL2ind seperately
Policy=zeros(6,N_a,N_semiz,N_j,'gpuArray'); % 1=d1, 2=d2, 3=d3, 4=joint(a1prime midpoint,a2prime), 5=a1prime L2
Policy(6,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
% For the return function we just want the full d=(d1,d2,d3) grid (used in the no-EV section which vectorises over d3)
n_d23=[n_d2,n_d3];
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)
a2ind=gpuArray(0:1:N_a2-1)';

% lowmemory indexing helpers
aind=gpuArray(0:1:N_a-1);
if vfoptions.lowmemory==0
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
elseif vfoptions.lowmemory==1
    special_n_semiz=ones(1,length(n_semiz));
end

% n-Monotonicity over a1
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% GI grid
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Preallocate (for the EV sections, which loop over d3)
V_ford3_hat=zeros(N_a,N_semiz,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz,N_d3,'gpuArray'); % beta-value gathered at the hat argmax, per d3
d12_ford3_hat=zeros(N_a,N_semiz,N_d3,'gpuArray');
joint_ford3_hat=zeros(N_a,N_semiz,N_d3,'gpuArray');
L2a1_ford3_hat=zeros(N_a,N_semiz,N_d3,'gpuArray');
L2flag_ford3_hat=2*ones(N_a,N_semiz,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No continuation value: return does not depend on the a3 dynamics, so vectorise over the full d=(d1,d2,d3)
    if vfoptions.lowmemory==0
        midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_semiz, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*semizBind;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy(3,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy(4,:,:,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
        Policy(5,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        Policy(6,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy(2,:,z_c,N_j)=ceil(d12_ind/N_d1); % d2
            Policy(3,:,z_c,N_j)=ceil(d_ind/N_d12); % d3
            Policy(4,:,z_c,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy(5,:,z_c,N_j)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            Policy(6,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end


    % Terminal period has no continuation, so Vunderbar coincides with Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]); % [N_a,N_semiz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_semiz]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_full=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % aprime depends on the CURRENT semiz, so (unlike the plain-expasset SemiExo version)
    % the interpolation cannot be hoisted out of the d3 loop: EVpre must be contracted over
    % the shock-prime index first (that contraction depends on d3 via pi_semiz), and only
    % then interpolated. See the d3 loops below.
    shock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EVc=EVpre.*shiftdim(pi_semiz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_semiz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta0beta is applied at the use sites and the undiscounted EVfine is reused for the Vunderbar gather
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(beta0beta*entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+beta0beta*entireEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            EVfine=reshape(entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;

            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);
            d12_ford3_hat(:,:,d3_c)=shiftdim(d_ind,1);
            joint_ford3_hat(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_hat(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the undiscounted interpolated continuation gathered at the hat argmax
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,N_j);
            EVc=EVpre.*shiftdim(pi_semiz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_semiz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta0beta is applied at the use sites and the undiscounted EVfine is reused for the Vunderbar gather
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(beta0beta*entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*entireEV_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;

                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_hat(:,z_c,d3_c)=shiftdim(d_ind,1);
                joint_ford3_hat(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_hat(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_hat(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the undiscounted interpolated continuation gathered at the hat argmax
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                V_ford3_under(:,z_c,d3_c)=V_ford3_hat(:,z_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,d3_max]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(d3_max,-1); % d3 is just the maximising index
    M=N_a*N_semiz;
    d3_max_lin=reshape(d3_max,[M,1]);
    idx=(1:M)'+M*(d3_max_lin-1);
    d12sel=reshape(d12_ford3_hat(idx),[1,N_a,N_semiz]);
    Policy(1,:,:,N_j)=rem(d12sel-1,N_d1)+1; % d1
    Policy(2,:,:,N_j)=ceil(d12sel/N_d1); % d2
    Policy(4,:,:,N_j)=reshape(joint_ford3_hat(idx),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(5,:,:,N_j)=reshape(L2a1_ford3_hat(idx),[1,N_a,N_semiz]); % a1primeL2ind
    Policy(6,:,:,N_j)=reshape(L2flag_ford3_hat(idx),[1,N_a,N_semiz]);
    % Vunderbar at the d3 chosen by the hat max
    Vunderbar(:,:,N_j)=reshape(V_ford3_under(idx),[N_a,N_semiz]);
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
    beta0beta=beta0*beta;

    EVpre=Vunderbar(:,:,jj+1); % [N_a,N_semiz]  -- continuation is Vunderbar (the beta-discounted value at the hat argmax)

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a3, n_semiz, d2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex, a3primeProbs are [N_d2,N_a3,N_semiz], indexed by the CURRENT semiz

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_semiz]
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs_full=repmat(a3primeProbs,N_a1*N_a2,1,1);
    % aprime depends on the CURRENT semiz, so (unlike the plain-expasset SemiExo version)
    % the interpolation cannot be hoisted out of the d3 loop: EVpre must be contracted over
    % the shock-prime index first (that contraction depends on d3 via pi_semiz), and only
    % then interpolated. See the d3 loops below.
    shock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EVc=EVpre.*shiftdim(pi_semiz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_semiz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta0beta is applied at the use sites and the undiscounted EVfine is reused for the Vunderbar gather
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(beta0beta*entireEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+beta0beta*entireEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
            EVfine=reshape(entireEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_semiz]);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz])+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;

            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);
            d12_ford3_hat(:,:,d3_c)=shiftdim(d_ind,1);
            joint_ford3_hat(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_hat(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz]);
            linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_hat(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            % Vunderbar: the undiscounted interpolated continuation gathered at the hat argmax
            linidx=reshape(maxindexL2,[1,N_a*N_semiz])+size(EVfine,1)*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            V_ford3_under(:,:,d3_c)=V_ford3_hat(:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_semiz=pi_semiz_J(:,:,d3_c,jj);
            EVc=EVpre.*shiftdim(pi_semiz',-1); % [N_a,shockprime,shock]
            EVc(isnan(EVc))=0;
            EV_2D=reshape(sum(EVc,2),[N_a,N_semiz]); % [aprime, CURRENT shock]
            Vlower=EV_2D(aprimeIndex+shock_offset);
            Vupper=EV_2D(aprimeplus1Index+shock_offset);
            aprimeProbs=aprimeProbs_full;
            aprimeProbs(Vlower==Vupper)=0; % skip interpolation where upper==lower
            EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_semiz]
            entireEV=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_semiz]); % undiscounted; beta0beta is applied at the use sites and the undiscounted EVfine is reused for the Vunderbar gather
            entireEVinterp=permute(interp1(a1_grid,permute(entireEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                entireEV_z=entireEV(:,:,:,:,:,:,z_c);
                entireEVinterp_z=entireEVinterp(:,:,:,:,:,:,z_c);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(beta0beta*entireEV_z,N_d1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_z+beta0beta*entireEV_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                EVfine=reshape(entireEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a]);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a])+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;

                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_hat(:,z_c,d3_c)=shiftdim(d_ind,1);
                joint_ford3_hat(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_hat(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_hat(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
                % Vunderbar: the undiscounted interpolated continuation gathered at the hat argmax
                linidx=reshape(maxindexL2,[1,N_a])+size(EVfine,1)*(0:N_a-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                V_ford3_under(:,z_c,d3_c)=V_ford3_hat(:,z_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,d3_max]=max(V_ford3_hat,[],3); % max over d3
    Vhat(:,:,jj)=V_jj;
    Policy(3,:,:,jj)=shiftdim(d3_max,-1); % d3 is just the maximising index
    M=N_a*N_semiz;
    d3_max_lin=reshape(d3_max,[M,1]);
    idx=(1:M)'+M*(d3_max_lin-1);
    d12sel=reshape(d12_ford3_hat(idx),[1,N_a,N_semiz]);
    Policy(1,:,:,jj)=rem(d12sel-1,N_d1)+1; % d1
    Policy(2,:,:,jj)=ceil(d12sel/N_d1); % d2
    Policy(4,:,:,jj)=reshape(joint_ford3_hat(idx),[1,N_a,N_semiz]); % joint(a1prime midpoint,a2prime)
    Policy(5,:,:,jj)=reshape(L2a1_ford3_hat(idx),[1,N_a,N_semiz]); % a1primeL2ind
    Policy(6,:,:,jj)=reshape(L2flag_ford3_hat(idx),[1,N_a,N_semiz]);
    % Vunderbar at the d3 chosen by the hat max
    Vunderbar(:,:,jj)=reshape(V_ford3_under(idx),[N_a,N_semiz]);
end


%% With grid interpolation, switch the a1prime part of the joint from midpoint to lower grid index
% Policy(4,:) is joint(a1prime midpoint,a2prime) and Policy(5,:) the second layer (ranges -n2short-1:1:1+n2short).
adjust=(Policy(5,:,:,:)<1+n2short+1);
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust; % a1prime part of joint -> lower grid point
Policy(5,:,:,:)=Policy(5,:,:,:)-(n2short+1)*(~adjust);

end
