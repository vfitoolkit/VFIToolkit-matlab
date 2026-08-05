function [V,Policy4]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC2A_GI2A_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, u_gridvals, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% SemiExo graft of ValueFnIter_FHorz_ExpAsset_DC2A_GI2A_raw (no e shock), keeping the divide-and-conquer + grid-interp math on a1.
% d1 is any other decision, d2 determines experience asset (a3), d3 determines semi-exog state (semiz).
% a1 is divide-conquered+grid-interp standard asset; a2 is a folded standard asset (choice a2prime); a3 is the experience asset.
% z is exogenous Markov, semiz is semi-exogenous; bothz=(semiz,z) with semiz varying fastest.
% Policy4 is 5-channel: 1=d1, 2=d2, 3=d3, 4=joint(a1prime midpoint->lower grid point after post-process, a2prime), 5=a1prime L2; PolicyL2flag appended as 6th.
% lowmemory: 2 shocks {z,semiz} => levels {0,1,2}.
%   =0 vectorise bothz; =1 split: outer-loop z / semiz parallel; =2 joint: loop over bothz.

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_d=N_d1*N_d2*N_d3;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_u=prod(n_u);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d1,d2,d3,joint(a1prime(mid),a2prime),a1primeL2ind seperately
Policy4=zeros(5,N_a,N_bothz,N_j,'gpuArray'); % 1=d1, 2=d2, 3=d3, 4=joint(a1prime midpoint,a2prime), 5=a1prime L2
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
% For the return function we just want the full d=(d1,d2,d3) grid (used in the no-EV section which vectorises over d3)
n_d23=[n_d2,n_d3];
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

d2ind_vec=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]; maps d12-index to d2-component (used inside the d3 loop where d=d12)
a2ind=gpuArray(0:1:N_a2-1)';

% lowmemory indexing helpers
aind=gpuArray(0:1:N_a-1);
if vfoptions.lowmemory==0
    bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
elseif vfoptions.lowmemory==1
    special_n_semiz=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
    semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
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
V_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
d12_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
joint_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
L2a1_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
L2flag_ford3_jj=2*ones(N_a,N_bothz,N_d3,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put u into third dimension
%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    % No continuation value: return does not depend on the a3 dynamics, so vectorise over the full d=(d1,d2,d3)
    if vfoptions.lowmemory==0
        midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');

        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
            end
        end

        midpoint=max(min(midpoint,N_a1-1),2);
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, n_bothz, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);

        d_ind        =rem(maxindexL2-1,N_d)+1;
        maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
        maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

        allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*bothzBind;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy4(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy4(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy4(3,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy4(4,:,:,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
        Policy4(5,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*bothzBind;
        linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*bothzBind;
        isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
        isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
        inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
        inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
        PolicyL2flag(1,:,:,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for z_c=1:N_z
            zind=(1:1:N_semiz)+N_semiz*(z_c-1);
            z_val=bothz_gridvals_J(zind,:,N_j);
            midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_semiz, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,zind,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind + N_d*N_a2*N_a*semizBind;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy4(1,:,zind,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy4(2,:,zind,N_j)=ceil(d12_ind/N_d1); % d2
            Policy4(3,:,zind,N_j)=ceil(d_ind/N_d12); % d3
            Policy4(4,:,zind,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4(5,:,zind,N_j)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind + N_d*n2long*N_a2*N_a*semizBind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,zind,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            midpoint=zeros(N_d,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            midpoint(:,1,:,level1ii,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                    [~,maxindex_inner]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d23, n_a2, n_a3, special_n_bothz, d_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);

            d_ind        =rem(maxindexL2-1,N_d)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d*n2long))+1;

            allind=d_ind + N_d*(maxindexL2a2-1) + N_d*N_a2*aind;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy4(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy4(2,:,z_c,N_j)=ceil(d12_ind/N_d1); % d2
            Policy4(3,:,z_c,N_j)=ceil(d_ind/N_d12); % d3
            Policy4(4,:,z_c,N_j)=midpoint(allind)+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy4(5,:,z_c,N_j)=maxindexL2a1; % a1primeL2ind

            linidx_lower=d_ind                  + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            linidx_upper=d_ind + N_d*(n2long-1) + N_d*n2long*(maxindexL2a2-1) + N_d*n2long*N_a2*aind;
            isInfLower=(ReturnMatrix_ii(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            PolicyL2flag(1,:,z_c,N_j)=2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;

            V_ford3_jj(:,:,d3_c)=shiftdim(Vtempii,1);
            d12_ford3_jj(:,:,d3_c)=shiftdim(d_ind,1);
            joint_ford3_jj(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_jj(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_bothz]);
            linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
            linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_jj(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,zind);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,zind);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z(d2aprimez);
                        [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;

                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_jj(:,zind,d3_c)=shiftdim(d_ind,1);
                joint_ford3_jj(:,zind,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_jj(:,zind,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_jj(:,zind,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z,N_d1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;

                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_jj(:,z_c,d3_c)=shiftdim(d_ind,1);
                joint_ford3_jj(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_jj(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_jj(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,d3_max]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,N_j)=V_jj;
    Policy4(3,:,:,N_j)=shiftdim(d3_max,-1); % d3 is just the maximising index
    M=N_a*N_bothz;
    d3_max_lin=reshape(d3_max,[M,1]);
    idx=(1:M)'+M*(d3_max_lin-1);
    d12sel=reshape(d12_ford3_jj(idx),[1,N_a,N_bothz]);
    Policy4(1,:,:,N_j)=rem(d12sel-1,N_d1)+1; % d1
    Policy4(2,:,:,N_j)=ceil(d12sel/N_d1); % d2
    Policy4(4,:,:,N_j)=reshape(joint_ford3_jj(idx),[1,N_a,N_bothz]); % joint(a1prime midpoint,a2prime)
    Policy4(5,:,:,N_j)=reshape(L2a1_ford3_jj(idx),[1,N_a,N_bothz]); % a1primeL2ind
    PolicyL2flag(1,:,:,N_j)=reshape(L2flag_ford3_jj(idx),[1,N_a,N_bothz]);
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=V(:,:,jj+1); % [N_a,N_bothz]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a3, n_u, d2_gridvals, a3_grid, u_gridvals, aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_bothz);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_u,N_bothz]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV_aprime=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % [N_d2*N_a1*N_a2,N_a3,N_bothz], indexed by bothzprime (d3-independent)
    EV_aprime=squeeze(sum((EV_aprime.*pi_u),3)); % integrate out u -> [N_d2*N_a1*N_a2,N_a3,zprime]
    EV_aprime(isnan(EV_aprime))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);
            midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV,N_d1,1,1,1,1,1,1);
            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
                    entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV(d2aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, n_bothz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprimez=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp(aprimez),[N_d12*n2long*N_a2,N_a,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

            d_ind        =rem(maxindexL2-1,N_d12)+1;
            maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
            maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
            allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*bothzBind;

            V_ford3_jj(:,:,d3_c)=shiftdim(Vtempii,1);
            d12_ford3_jj(:,:,d3_c)=shiftdim(d_ind,1);
            joint_ford3_jj(:,:,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
            L2a1_ford3_jj(:,:,d3_c)=shiftdim(maxindexL2a1,1);

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_bothz]);
            linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
            linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
            inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
            inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
            L2flag_ford3_jj(:,:,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
        end

    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,zind);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,zind);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_z,N_d1,1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_d3,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                        entireRHS_ii_d3=ReturnMatrix_ii_d3+DiscountedEV_z(d2aprimez);
                        [~,maxindex_inner]=max(entireRHS_ii_d3,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_semiz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1fine*N_a2*N_a3*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii_d3=reshape(ReturnMatrix_ii_d3+DiscountedEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind + N_d12*N_a2*N_a*semizBind;

                V_ford3_jj(:,zind,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_jj(:,zind,d3_c)=shiftdim(d_ind,1);
                joint_ford3_jj(:,zind,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_jj(:,zind,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_d3,[N_d12*n2long*N_a2,N_a,N_semiz]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind + N_d12*n2long*N_a2*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_jj(:,zind,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals=[d12_gridvals,d3_grid(d3_c).*ones(N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            EV=EV_aprime.*shiftdim(pi_bothz',-2);
            EV(isnan(EV))=0;
            EV=squeeze(sum(EV,3));
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_z=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                midpoint=zeros(N_d12,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z,N_d1,1,1,1,1,1);
                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                midpoint(:,1,:,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                        d2aprime=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z(d2aprime);
                        [~,maxindex_inner]=max(entireRHS_ii_z,[],2);
                        midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:);
                        midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, [n_d2,1], n_a2, n_a3, special_n_bothz, d123_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime_z=d2ind_vec + N_d2*(a1primeindexesfine-1) + N_d2*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1fine*N_a2*shiftdim((0:1:N_a3-1),-4);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_z(aprime_z),[N_d12*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);

                d_ind        =rem(maxindexL2-1,N_d12)+1;
                maxindexL2a1 =rem(floor((maxindexL2-1)/N_d12),n2long)+1;
                maxindexL2a2 =floor((maxindexL2-1)/(N_d12*n2long))+1;
                allind=d_ind + N_d12*(maxindexL2a2-1) + N_d12*N_a2*aind;

                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d12_ford3_jj(:,z_c,d3_c)=shiftdim(d_ind,1);
                joint_ford3_jj(:,z_c,d3_c)=shiftdim(midpoint(allind)+N_a1*(maxindexL2a2-1),1);
                L2a1_ford3_jj(:,z_c,d3_c)=shiftdim(maxindexL2a1,1);

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii_z,[N_d12*n2long*N_a2,N_a]);
                linidx_lower=d_ind                    + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                linidx_upper=d_ind + N_d12*(n2long-1) + N_d12*n2long*(maxindexL2a2-1) + N_d12*n2long*N_a2*aind;
                isInfLower=(ReturnMatrix_ii_flat(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_flat(linidx_upper)==-Inf);
                inLowerStrict=(maxindexL2a1>=2)         & (maxindexL2a1<=n2short+1);
                inUpperStrict=(maxindexL2a1>=n2short+3) & (maxindexL2a1<=n2long-1);
                L2flag_ford3_jj(:,z_c,d3_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,d3_max]=max(V_ford3_jj,[],3); % max over d3
    V(:,:,jj)=V_jj;
    Policy4(3,:,:,jj)=shiftdim(d3_max,-1); % d3 is just the maximising index
    M=N_a*N_bothz;
    d3_max_lin=reshape(d3_max,[M,1]);
    idx=(1:M)'+M*(d3_max_lin-1);
    d12sel=reshape(d12_ford3_jj(idx),[1,N_a,N_bothz]);
    Policy4(1,:,:,jj)=rem(d12sel-1,N_d1)+1; % d1
    Policy4(2,:,:,jj)=ceil(d12sel/N_d1); % d2
    Policy4(4,:,:,jj)=reshape(joint_ford3_jj(idx),[1,N_a,N_bothz]); % joint(a1prime midpoint,a2prime)
    Policy4(5,:,:,jj)=reshape(L2a1_ford3_jj(idx),[1,N_a,N_bothz]); % a1primeL2ind
    PolicyL2flag(1,:,:,jj)=reshape(L2flag_ford3_jj(idx),[1,N_a,N_bothz]);
end


%% With grid interpolation, switch the a1prime part of the joint from midpoint to lower grid index
% Policy4(4,:) is joint(a1prime midpoint,a2prime) and Policy4(5,:) the second layer (ranges -n2short-1:1:1+n2short).
adjust=(Policy4(5,:,:,:)<1+n2short+1);
Policy4(4,:,:,:)=Policy4(4,:,:,:)-adjust; % a1prime part of joint -> lower grid point
Policy4(5,:,:,:)=adjust.*Policy4(5,:,:,:)+(1-adjust).*(Policy4(5,:,:,:)-n2short-1);

Policy4=[Policy4;PolicyL2flag];

end
