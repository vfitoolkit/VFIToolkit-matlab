function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_SemiExo_DC1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z (only semiz)
% Policy output has the choices on the first dimension: (d1,d2,d3,d4,a1prime).
%
% Epstein-Zin graft onto ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_noz_raw:
% V' is transformed by ^ezc5 (masked) once per age before the expectations
% (joint certainty-equivalent over (u,semiz'): all sums are on the transformed
% object); temp4 (post-certainty-equivalent continuation) is refined over d2
% using ezc9*max(ezc9*ezc3*.) and indexed exactly where the vNM code indexes
% DiscountedEV; the ^ezc7 mask wraps each level's entireRHS before its max (a
% monotone transform, so the divide-and-conquer monotonicity logic is unaffected).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(5,N_a,N_semiz,N_j,'gpuArray'); % d1, d2, d3, d4, a1prime

%%
u_grid=gpuArray(u_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));
d1d3d4a1_gridvals=gpuArray(CreateGridvals([n_d1,n_d3,n_d4,n_a1],[d1_grid;d3_grid;d4_grid;a1_grid],1));
a1a2_gridvals=gpuArray(CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2Bind=gpuArray(0:1:N_a2-1);
zBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
d3ind=repelem((1:1:N_d3)',N_d1,1);
aind=gpuArray(0:1:N_a-1);
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

% Accumulators for the choice of d4 (max is taken across d4 at the end of each period)
V_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d1_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d2_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
d3_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');
a1prime_ford4_jj=zeros(N_a,N_semiz,N_d4,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec); % This depends on a2prime
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    WGmatrix=repelem(WGmatrix,N_a1,1); % expand from a2prime to (a1prime,a2prime) [warm-glow does not depend on a1prime]

    % Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d23*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d23*N_a1,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d23*N_a1,N_u]

    % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
    % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
    skipinterp=logical(WGmatrix(aprimeIndex)==WGmatrix(aprimeplus1Index));
    aprimeProbs(skipinterp)=0;

    WG1=reshape(WGmatrix(aprimeIndex),[N_d23*N_a1,N_u]).*aprimeProbs; % probability of lower grid point
    WG2=reshape(WGmatrix(aprimeplus1Index),[N_d23*N_a1,N_u]).*(1-aprimeProbs); % probability of upper grid point
    % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
    WG1(isnan(WG1))=0;
    WG2(isnan(WG2))=0;
    % Expectation over u (using pi_u), and then add the lower and upper
    WGmatrix=sum((WG1.*pi_u_col'),2)+sum((WG2.*pi_u_col'),2); % [N_d23*N_a1,1], sum over u

    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
    % Now just make it the right shape (needs to broadcast against temp4, which spans semiz when semiz is vectorized)
    if isfield(vfoptions,'V_Jplus1')
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,N_semiz);
        end
    end
else
    WGmatrix=0;
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix).*(ReturnMatrix~=0)); % finite and not zero
        ReturnMatrix(becareful)=(ezc1*ReturnMatrix(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix(ReturnMatrix==0)=-Inf;

        if warmglow==1
            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(ezc9*reshape((~isinf(ReturnMatrix)).*ReturnMatrix,[N_d1,N_d3*N_d4*N_a1,N_a,N_semiz]),[],1);
            % Second: warm-glow, we can refine out d2
            [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
            % WGmatrix_onlyd3 is over (d3,a1prime); rows of ReturnMatrix_onlyd3 are (d3,d4,a1prime), so spread it over d4
            WGmatrix_onlyd3=reshape(repmat(reshape(WGmatrix_onlyd3,[N_d3,1,N_a1]),1,N_d4,1),[1,N_d3*N_d4*N_a1]);
            % Now put together entireRHS, which just depends on (d3,d4,a1prime)
            entireRHS=shiftdim(ezc9*ReturnMatrix_onlyd3+ezc9*WGmatrix_onlyd3,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d3*N_d4)+1;
            d3index=rem(dindex-1,N_d3)+1;
            a1primeindex=ceil(maxindex/(N_d3*N_d4));
            Policy(1,:,:,N_j)=d1index(maxindex+N_d3*N_d4*N_a1*aind+N_d3*N_d4*N_a1*N_a*semizind); % d1
            Policy(2,:,:,N_j)=d2index(d3index+N_d3*(a1primeindex-1)); % d2 (note: no a nor semiz in WGmatrix)
            Policy(3,:,:,N_j)=d3index; % d3
            Policy(4,:,:,N_j)=ceil(dindex/N_d3); % d4
            Policy(5,:,:,N_j)=a1primeindex; % a1prime
        elseif warmglow==0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix,[],1);
            V(:,:,N_j)=shiftdim(Vtemp,1);
            dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
            d1d3_ind=rem(dindex-1,N_d13)+1;
            d1part=rem(d1d3_ind-1,N_d1)+1;
            d3part=ceil(d1d3_ind/N_d1);
            d4part=ceil(dindex/N_d13);
            a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
            Policy(1,:,:,N_j)=d1part;
            Policy(2,:,:,N_j)=1; % d2 is meaningless in the terminal period
            Policy(3,:,:,N_j)=d3part;
            Policy(4,:,:,N_j)=d4part;
            Policy(5,:,:,N_j)=a1primepart;
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d3,n_d4,n_a1], [n_a1,n_a2], special_n_semiz, d1d3d4a1_gridvals, a1a2_gridvals, z_val, ReturnFnParamsVec);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_z).*(ReturnMatrix_z~=0)); % finite and not zero
            ReturnMatrix_z(becareful)=(ezc1*ReturnMatrix_z(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
            ReturnMatrix_z(ReturnMatrix_z==0)=-Inf;

            if warmglow==1
                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                [ReturnMatrix_onlyd3,d1index]=max(ezc9*reshape((~isinf(ReturnMatrix_z)).*ReturnMatrix_z,[N_d1,N_d3*N_d4*N_a1,N_a]),[],1);
                % Second: warm-glow, we can refine out d2
                [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
                % WGmatrix_onlyd3 is over (d3,a1prime); rows of ReturnMatrix_onlyd3 are (d3,d4,a1prime), so spread it over d4
                WGmatrix_onlyd3=reshape(repmat(reshape(WGmatrix_onlyd3,[N_d3,1,N_a1]),1,N_d4,1),[1,N_d3*N_d4*N_a1]);
                % Now put together entireRHS, which just depends on (d3,d4,a1prime)
                entireRHS=shiftdim(ezc9*ReturnMatrix_onlyd3+ezc9*WGmatrix_onlyd3,1);

                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d3*N_d4)+1;
                d3index=rem(dindex-1,N_d3)+1;
                a1primeindex=ceil(maxindex/(N_d3*N_d4));
                Policy(1,:,z_c,N_j)=d1index(maxindex+N_d3*N_d4*N_a1*aind); % d1
                Policy(2,:,z_c,N_j)=d2index(d3index+N_d3*(a1primeindex-1)); % d2 (note: no a nor semiz in WGmatrix)
                Policy(3,:,z_c,N_j)=d3index; % d3
                Policy(4,:,z_c,N_j)=ceil(dindex/N_d3); % d4
                Policy(5,:,z_c,N_j)=a1primeindex; % a1prime
            elseif warmglow==0
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
                V(:,z_c,N_j)=shiftdim(Vtemp,1);
                dindex=rem(maxindex-1,N_d1*N_d3*N_d4)+1;
                d1d3_ind=rem(dindex-1,N_d13)+1;
                d1part=rem(d1d3_ind-1,N_d1)+1;
                d3part=ceil(d1d3_ind/N_d1);
                d4part=ceil(dindex/N_d13);
                a1primepart=ceil(maxindex/(N_d1*N_d3*N_d4));
                Policy(1,:,z_c,N_j)=d1part;
                Policy(2,:,z_c,N_j)=1; % d2 is meaningless in the terminal period
                Policy(3,:,z_c,N_j)=d3part;
                Policy(4,:,z_c,N_j)=d4part;
                Policy(5,:,z_c,N_j)=a1primepart;
            end
        end
    end
else
    % Using V_Jplus1
    if warmglow==0 % if warmglow==1 these were already created above
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);
    end

    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    % Part of Epstein-Zin is before taking expectation
    temp=EVnext;
    temp(isfinite(EVnext))=(ezc4*EVnext(isfinite(EVnext))).^ezc5(N_j);
    temp(EVnext==0)=0; % otherwise zero to negative power is set to infinity

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV==0)=0;
            end

            % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_semiz]),[],1);
            temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            % DiscountedEV
            DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
            entireRHS_ii=ezc1*RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d1part=rem(pol_d13_a1-1,N_d1)+1;
            d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
            a1primepart=ceil(pol_d13_a1/N_d13);
            zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d1_ford4_jj(curraindex,:,d4_c)=d1part;
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                % Part of Epstein-Zin is after taking expectation
                temp4=EV_z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_z==0)=0;
                end

                % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
                [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
                temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1]);

                % DiscountedEV
                DiscountedEV_z=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii_z).*(ReturnMatrix_ii_z~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii_z;
                temp2_ii(becareful)=ReturnMatrix_ii_z(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii_z==0)=-Inf;
                RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_z=ezc1*RM+DEV;
                entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,N_a1,vfoptions.level1n,N_a2]);
                temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                lin=d3part+N_d3*(a1primepart-1);
                d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_z=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                        entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(N_j);
                        entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_z=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                        entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(N_j);
                        entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    linidx_d4=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(reshape(d4winner,[N_a*N_semiz,1])-1);
    Policy(1,:,:,N_j)=reshape(d1_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(2,:,:,N_j)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(3,:,:,N_j)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz]);
    Policy(5,:,:,N_j)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
end


%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        WGmatrix=repelem(WGmatrix,N_a1,1); % expand from a2prime to (a1prime,a2prime) [warm-glow does not depend on a1prime]

        % Switch WGmatrix from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
        % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
        skipinterp=logical(WGmatrix(aprimeIndex)==WGmatrix(aprimeplus1Index));
        aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d23*N_a1,N_u]
        aprimeProbs(skipinterp)=0;

        WG1=reshape(WGmatrix(aprimeIndex),[N_d23*N_a1,N_u]).*aprimeProbs; % probability of lower grid point
        WG2=reshape(WGmatrix(aprimeplus1Index),[N_d23*N_a1,N_u]).*(1-aprimeProbs); % probability of upper grid point
        % If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
        WG1(isnan(WG1))=0;
        WG2(isnan(WG2))=0;
        % Expectation over u (using pi_u), and then add the lower and upper
        WGmatrix=sum((WG1.*pi_u_col'),2)+sum((WG2.*pi_u_col'),2); % [N_d23*N_a1,1], sum over u
        % Now just make it the right shape (needs to broadcast against temp4, which spans semiz when semiz is vectorized)
        if vfoptions.lowmemory==0
            WGmatrix=WGmatrix.*ones(1,N_semiz);
        end
    end

    EVnext=V(:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVnext;
    temp(isfinite(EVnext))=(ezc4*EVnext(isfinite(EVnext))).^ezc5(jj);
    temp(EVnext==0)=0;

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c); % no kron in noz case
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=temp.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a2primeProbs,N_a1,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a1,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a1,N_semiz]);

            % Part of Epstein-Zin is after taking expectation
            temp4=EV;
            if warmglow==1
                becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV==0)&(WGmatrix==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV==0)=0;
            end

            % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
            [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_semiz]),[],1);
            temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_semiz]);

            % DiscountedEV
            DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_semiz]);

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,n_semiz, d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
            temp2_ii=ReturnMatrix_ii;
            temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii==0)=-Inf;
            RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_semiz]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_semiz]);
            entireRHS_ii=ezc1*RM+DEV;
            entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_semiz]);
            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_semiz]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d1part=rem(pol_d13_a1-1,N_d1)+1;
            d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
            a1primepart=ceil(pol_d13_a1/N_d13);
            zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            d1_ford4_jj(curraindex,:,d4_c)=d1part;
            d3_ford4_jj(curraindex,:,d4_c)=d3part;
            a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
            d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,n_semiz, d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_semiz]);
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford4_jj(curraindex,:,d4_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d1part=rem(pol_d13_a1-1,N_d1)+1;
                    d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    zidx=repmat(gpuArray(1:N_semiz),size(pol_d13_a1,1),1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    d1_ford4_jj(curraindex,:,d4_c)=d1part;
                    d3_ford4_jj(curraindex,:,d4_c)=d3part;
                    a1prime_ford4_jj(curraindex,:,d4_c)=a1primepart;
                    d2_ford4_jj(curraindex,:,d4_c)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);

                EV_z=temp.*pi_semizd4(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a2primeProbs,N_a1,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a1,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a1,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                % Part of Epstein-Zin is after taking expectation
                temp4=EV_z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_z==0)=0;
                end

                % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
                [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1]),[],1);
                temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1]);

                % DiscountedEV
                DiscountedEV_z=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1]);

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],n_a1,vfoptions.level1n,n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii_z).*(ReturnMatrix_ii_z~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii_z;
                temp2_ii(becareful)=ReturnMatrix_ii_z(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii_z==0)=-Inf;
                RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_z=ezc1*RM+DEV;
                entireRHS_ii_z=reshape(entireRHS_ii_z,[N_d13,N_a1,vfoptions.level1n,N_a2]);
                temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
                entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1);
                d1part=rem(pol_d13_a1-1,N_d1)+1;
                d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                a1primepart=ceil(pol_d13_a1/N_d13);
                lin=d3part+N_d3*(a1primepart-1);
                d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],maxgap(ii)+1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_z=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                        entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(jj);
                        entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d3,special_n_d4],1,level1iidiff(ii),n_a2,ones(1,length(n_semiz)), d13_with_d4, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_z=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_z).*(entireRHS_ii_z~=0));
                        entireRHS_ii_z(temp5)=entireRHS_ii_z(temp5).^ezc7(jj);
                        entireRHS_ii_z(entireRHS_ii_z==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford4_jj(curraindex,z_c,d4_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                        d1part=rem(pol_d13_a1-1,N_d1)+1;
                        d3part=rem(ceil(pol_d13_a1/N_d1)-1,N_d3)+1;
                        a1primepart=ceil(pol_d13_a1/N_d13);
                        lin=d3part+N_d3*(a1primepart-1);
                        d1_ford4_jj(curraindex,z_c,d4_c)=d1part;
                        d3_ford4_jj(curraindex,z_c,d4_c)=d3part;
                        a1prime_ford4_jj(curraindex,z_c,d4_c)=a1primepart;
                        d2_ford4_jj(curraindex,z_c,d4_c)=d2index_z(lin);
                    end
                end
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,jj)=Vbest;
    linidx_d4=(1:1:N_a*N_semiz)'+(N_a*N_semiz)*(reshape(d4winner,[N_a*N_semiz,1])-1);
    Policy(1,:,:,jj)=reshape(d1_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(2,:,:,jj)=reshape(d2_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(3,:,:,jj)=reshape(d3_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
    Policy(4,:,:,jj)=reshape(d4winner,[1,N_a,N_semiz]);
    Policy(5,:,:,jj)=reshape(a1prime_ford4_jj(linidx_d4),[1,N_a,N_semiz]);
end


end
