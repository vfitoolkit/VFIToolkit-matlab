function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_DC1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% e: iid start-of-period shock
%
% Epstein-Zin graft onto ValueFnIter_FHorz_RiskyAsset_DC1_e_raw: V' is
% transformed by ^ezc5 (masked) once per age before the expectations (one joint
% certainty-equivalent over (u,z',e'): the e-expectation, the z-expectation and
% the u-lottery are all taken on the transformed object); temp4
% (post-certainty-equivalent continuation) is refined over d2 using
% ezc9*max(ezc9*.) and indexed exactly where the vNM code indexes DiscountedEV;
% the ^ezc7 mask wraps each level's entireRHS before its max (a monotone
% transform, so the divide-and-conquer monotonicity logic is unaffected).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

% For ReturnFn (d1 and d3 only)
n_d13=[n_d1,n_d3];
N_d13=N_d1*N_d3;
d13_grid=[d1_grid;d3_grid];
% For aprimeFn (d2 and d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid; % already a column vector
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory==0
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
    eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=repelem((1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));
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
    WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % [N_d23*N_a1,1], sum over u

    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
else
    WGmatrix=zeros(N_d23*N_a1,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')

    % Refine d2 out of the (transformed) warm-glow term [WGmatrix is zeros if no warm-glow, so d2 policy is just 1]
    [WGmatrix_onlyd3,d2index]=max(ezc9*reshape((~isinf(WGmatrix)).*WGmatrix,[N_d2,N_d3*N_a1]),[],1);
    WGmatrix_onlyd3=ezc9*reshape(WGmatrix_onlyd3,[N_d3,N_a1]);
    d2index_resh=reshape(d2index,[N_d3,N_a1]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j); % Otherwise can get things like 0 to negative power equals infinity
        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;

        RM=reshape(ReturnMatrix_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);
        WGr=reshape(WGmatrix_onlyd3,[1,N_d3,N_a1]);
        entireRHS_ii=RM+WGr; % warm-glow (zero if not using)
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z,N_e]
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        d1part=rem(d_ind-1,N_d1)+1;
        d3part=ceil(d_ind/N_d1);
        a1primepart=ceil(pol_d13_a1/N_d13);
        Policy(1,curraindex,:,:,N_j)=d1part;
        Policy(3,curraindex,:,:,N_j)=d3part;
        Policy(4,curraindex,:,:,N_j)=a1primepart;
        % Get the d2Policy [note: no z nor e in WGmatrix]
        lin=d3part+N_d3*(a1primepart-1);
        Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprime=d3ind+N_d3*(a1primeindexes-1);
                entireRHS_ii=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1); % [npts,N_z,N_e]
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy [note: no z nor e in WGmatrix]
                lin=d3part+N_d3*(a1primepart-1);
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprime=d3ind+N_d3*(loweredge-1);
                entireRHS_ii=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13,level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy [note: no z nor e in WGmatrix]
                lin=d3part+N_d3*(a1primepart-1);
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite and not zero
            ReturnMatrix_ii_e(becareful)=(ezc1*ReturnMatrix_ii_e(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_ii_e(ReturnMatrix_ii_e==0)=-Inf;

            RM=reshape(ReturnMatrix_ii_e,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z]);
            WGr=reshape(WGmatrix_onlyd3,[1,N_d3,N_a1]);
            entireRHS_ii_e=RM+WGr; % warm-glow (zero if not using)
            entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z]);

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z]
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            d1part=rem(d_ind-1,N_d1)+1;
            d3part=ceil(d_ind/N_d1);
            a1primepart=ceil(pol_d13_a1/N_d13);
            Policy(1,curraindex,:,e_c,N_j)=d1part;
            Policy(3,curraindex,:,e_c,N_j)=d3part;
            Policy(4,curraindex,:,e_c,N_j)=a1primepart;
            % Get the d2Policy [note: no z in WGmatrix]
            lin=d3part+N_d3*(a1primepart-1);
            Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprime=d3ind+N_d3*(a1primeindexes-1);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy [note: no z in WGmatrix]
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprime=d3ind+N_d3*(loweredge-1);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13,level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy [note: no z in WGmatrix]
                    lin=d3part+N_d3*(a1primepart-1);
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii_ze).*(ReturnMatrix_ii_ze~=0)); % finite and not zero
                ReturnMatrix_ii_ze(becareful)=(ezc1*ReturnMatrix_ii_ze(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_ii_ze(ReturnMatrix_ii_ze==0)=-Inf;

                RM=reshape(ReturnMatrix_ii_ze,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                WGr=reshape(WGmatrix_onlyd3,[1,N_d3,N_a1]);
                entireRHS_ii_ze=RM+WGr; % warm-glow (zero if not using)
                entireRHS_ii_ze=reshape(entireRHS_ii_ze,[N_d13,N_a1,vfoptions.level1n,N_a2]);

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1); % [npts,1]
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                Policy(2,curraindex,z_c,e_c,N_j)=d2index_resh(d3part+N_d3*(a1primepart-1));

                % Divide-and-conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+WGmatrix_onlyd3(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    end
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                    Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                    Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                    Policy(2,curraindex,z_c,e_c,N_j)=d2index_resh(d3part+N_d3*(a1primepart-1));
                end
            end
        end
    end

else % V_Jplus1

    if warmglow==0 % if warmglow==1 these were already created above
        % Build a2primeIndex and a2primeProbs for RiskyAsset
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
        [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);
    end

    % Get EV in terms of next period endogenous states
    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);

    % Part of Epstein-Zin is before taking expectation
    temp=EVnext;
    temp(isfinite(EVnext))=(ezc4*EVnext(isfinite(EVnext))).^ezc5(N_j);
    temp(EVnext==0)=0; % otherwise zero to negative power is set to infinity

    % Take expectation over e
    temp=sum(temp.*shiftdim(pi_e_J(:,N_j+1),-2),3); % [N_a,N_z]

    EV=temp.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Part of Epstein-Zin is after taking expectation
    temp4=EV;
    if warmglow==1
        WGmatrixbig=WGmatrix.*ones(1,N_z);
        becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
        temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
        temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
        temp4(EV==0)=0;
    end

    % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
    [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_z]),[],1);
    temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;

        RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);
        DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_z,1]);
        entireRHS_ii=ezc1*RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);

        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z,N_e]
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        d1part=rem(d_ind-1,N_d1)+1;
        d3part=ceil(d_ind/N_d1);
        a1primepart=ceil(pol_d13_a1/N_d13);
        Policy(1,curraindex,:,:,N_j)=d1part;
        Policy(3,curraindex,:,:,N_j)=d3part;
        Policy(4,curraindex,:,:,N_j)=a1primepart;
        % Get the d2Policy via lookup on (d3,a1prime,z)
        [npts,nz,ne]=size(pol_d13_a1);
        zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d13_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z,N_e]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,N_j)=d1part;
                Policy(3,curraindex,:,:,N_j)=d3part;
                Policy(4,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d13_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,:,N_j)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite and not zero
            temp2_ii=ReturnMatrix_ii_e;
            temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_ii_e==0)=-Inf;

            RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_z]);
            entireRHS_ii_e=ezc1*RM+DEV;
            entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z]);

            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1); % [npts,N_z]
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            d1part=rem(d_ind-1,N_d1)+1;
            d3part=ceil(d_ind/N_d1);
            a1primepart=ceil(pol_d13_a1/N_d13);
            Policy(1,curraindex,:,e_c,N_j)=d1part;
            Policy(3,curraindex,:,e_c,N_j)=d3part;
            Policy(4,curraindex,:,e_c,N_j)=a1primepart;
            % Get the d2Policy
            [npts,nz]=size(pol_d13_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(N_j);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,N_j)=d1part;
                    Policy(3,curraindex,:,e_c,N_j)=d3part;
                    Policy(4,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(2,curraindex,:,e_c,N_j)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1]
            d2index_z=d2index_resh(:,:,z_c);          % [N_d3,N_a1]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii_ze).*(ReturnMatrix_ii_ze~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii_ze;
                temp2_ii(becareful)=ReturnMatrix_ii_ze(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_ii_ze==0)=-Inf;

                RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_ze=ezc1*RM+DEV;
                entireRHS_ii_ze=reshape(entireRHS_ii_ze,[N_d13,N_a1,vfoptions.level1n,N_a2]);

                temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(N_j);
                entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1); % [npts,1]
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                Policy(2,curraindex,z_c,e_c,N_j)=d2index_z(d3part+N_d3*(a1primepart-1));

                % Divide and conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_ze=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                        entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(N_j);
                        entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_ze=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                        entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(N_j);
                        entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    end
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,e_c,N_j)=d1part;
                    Policy(3,curraindex,z_c,e_c,N_j)=d3part;
                    Policy(4,curraindex,z_c,e_c,N_j)=a1primepart;
                    Policy(2,curraindex,z_c,e_c,N_j)=d2index_z(d3part+N_d3*(a1primepart-1));
                end
            end
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % Build a2primeIndex and a2primeProbs for RiskyAsset
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
        WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2); % [N_d23*N_a1,1], sum over u
    end

    % Get EV in terms of next period endogenous states
    EVpre=V(:,:,:,jj+1);

    % Part of Epstein-Zin is before taking expectation
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    % Take expectation over e
    temp=sum(temp.*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a,N_z]

    EV=temp.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);
    EV=reshape(EV,[N_a,N_z]);

    % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
    skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
    aprimeProbs=repmat(a2primeProbs,N_a1,N_z);
    aprimeProbs(skipinterp)=0;
    aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);
    % Take the expectation over the between period iid u shock
    EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbs;
    EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs);
    EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2);
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Part of Epstein-Zin is after taking expectation
    temp4=EV;
    if warmglow==1
        WGmatrixbig=WGmatrix.*ones(1,N_z);
        becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
        temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
        temp4((EV==0)&(WGmatrixbig==0))=0; % Is actually zero
    else % not using warmglow
        temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
        temp4(EV==0)=0;
    end

    % Refine d2 out of temp4 before combining with ReturnFn [ezc9 handles the sign for the max]
    [temp4_onlyd3,d2index]=max(ezc9*ezc3*reshape((~isinf(temp4)).*temp4,[N_d2,N_d3*N_a1,N_z]),[],1);
    temp4_onlyd3=reshape(temp4_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*ezc9*reshape(temp4_onlyd3,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
        temp2_ii=ReturnMatrix_ii;
        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
        temp2_ii(ReturnMatrix_ii==0)=-Inf;

        RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);
        DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_z,1]);
        entireRHS_ii=ezc1*RM+DEV;
        entireRHS_ii=reshape(entireRHS_ii,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z,N_e]);

        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        entireRHS_ii(entireRHS_ii==0)=-Inf;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        pol_d13_a1=shiftdim(maxindex2,1);
        d_ind=rem(pol_d13_a1-1,N_d13)+1;
        d1part=rem(d_ind-1,N_d1)+1;
        d3part=ceil(d_ind/N_d1);
        a1primepart=ceil(pol_d13_a1/N_d13);
        Policy(1,curraindex,:,:,jj)=d1part;
        Policy(3,curraindex,:,:,jj)=d3part;
        Policy(4,curraindex,:,:,jj)=a1primepart;
        % Get the d2Policy
        [npts,nz,ne]=size(pol_d13_a1);
        zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(2,curraindex,:,:,jj)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,jj)=d1part;
                Policy(3,curraindex,:,:,jj)=d3part;
                Policy(4,curraindex,:,:,jj)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d13_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,:,jj)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii;
                temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii==0)=-Inf;
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z,N_e]);
                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d13)+1);
                allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind+N_d13*N_a2*N_z*eBind;
                pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,:,:,jj)=d1part;
                Policy(3,curraindex,:,:,jj)=d3part;
                Policy(4,curraindex,:,:,jj)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d13_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(2,curraindex,:,:,jj)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);

            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_ii_e).*(ReturnMatrix_ii_e~=0)); % finite and not zero
            temp2_ii=ReturnMatrix_ii_e;
            temp2_ii(becareful)=ReturnMatrix_ii_e(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_ii_e==0)=-Inf;

            RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2,N_z]);
            DEV=reshape(DiscountedEV,[1,N_d3,N_a1,1,1,N_z]);
            entireRHS_ii_e=ezc1*RM+DEV;
            entireRHS_ii_e=reshape(entireRHS_ii_e,[N_d13,N_a1,vfoptions.level1n,N_a2,N_z]);

            temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
            entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
            entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d13*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            pol_d13_a1=shiftdim(maxindex2,1);
            d_ind=rem(pol_d13_a1-1,N_d13)+1;
            d1part=rem(d_ind-1,N_d1)+1;
            d3part=ceil(d_ind/N_d1);
            a1primepart=ceil(pol_d13_a1/N_d13);
            Policy(1,curraindex,:,e_c,jj)=d1part;
            Policy(3,curraindex,:,e_c,jj)=d3part;
            Policy(4,curraindex,:,e_c,jj)=a1primepart;
            % Get the d2Policy
            [npts,nz]=size(pol_d13_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,jj)=d1part;
                    Policy(3,curraindex,:,e_c,jj)=d3part;
                    Policy(4,curraindex,:,e_c,jj)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ezc1*temp2_ii+DiscountedEV(d3aprimez),[N_d13,level1iidiff(ii)*N_a2,N_z]);
                    temp5=logical(isfinite(entireRHS_ii_e).*(entireRHS_ii_e~=0));
                    entireRHS_ii_e(temp5)=entireRHS_ii_e(temp5).^ezc7(jj);
                    entireRHS_ii_e(entireRHS_ii_e==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d13)+1);
                    allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii))+N_d13*N_a2*zBind;
                    pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,:,e_c,jj)=d1part;
                    Policy(3,curraindex,:,e_c,jj)=d3part;
                    Policy(4,curraindex,:,e_c,jj)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d13_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(2,curraindex,:,e_c,jj)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1]
            d2index_z=d2index_resh(:,:,z_c);          % [N_d3,N_a1]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_ii_ze).*(ReturnMatrix_ii_ze~=0)); % finite and not zero
                temp2_ii=ReturnMatrix_ii_ze;
                temp2_ii(becareful)=ReturnMatrix_ii_ze(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_ii_ze==0)=-Inf;

                RM=reshape(temp2_ii,[N_d1,N_d3,N_a1,vfoptions.level1n,N_a2]);
                DEV=reshape(DiscountedEV_z,[1,N_d3,N_a1,1,1]);
                entireRHS_ii_ze=ezc1*RM+DEV;
                entireRHS_ii_ze=reshape(entireRHS_ii_ze,[N_d13,N_a1,vfoptions.level1n,N_a2]);

                temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(jj);
                entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d13*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                pol_d13_a1=shiftdim(maxindex2,1); % [npts,1]
                d_ind=rem(pol_d13_a1-1,N_d13)+1;
                d1part=rem(d_ind-1,N_d1)+1;
                d3part=ceil(d_ind/N_d1);
                a1primepart=ceil(pol_d13_a1/N_d13);
                Policy(1,curraindex,z_c,e_c,jj)=d1part;
                Policy(3,curraindex,z_c,e_c,jj)=d3part;
                Policy(4,curraindex,z_c,e_c,jj)=a1primepart;
                Policy(2,curraindex,z_c,e_c,jj)=d2index_z(d3part+N_d3*(a1primepart-1));

                % Divide and conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_ze=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                        entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(jj);
                        entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d13_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite and not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_ze=reshape(ezc1*temp2_ii+DiscountedEV_z(d3aprime),[N_d13,level1iidiff(ii)*N_a2]);
                        temp5=logical(isfinite(entireRHS_ii_ze).*(entireRHS_ii_ze~=0));
                        entireRHS_ii_ze(temp5)=entireRHS_ii_ze(temp5).^ezc7(jj);
                        entireRHS_ii_ze(entireRHS_ii_ze==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d13)+1);
                        allind=dind+N_d13*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d13_a1=shiftdim(maxindex+N_d13*(loweredge(allind)-1),1);
                    end
                    d_ind=rem(pol_d13_a1-1,N_d13)+1;
                    d1part=rem(d_ind-1,N_d1)+1;
                    d3part=ceil(d_ind/N_d1);
                    a1primepart=ceil(pol_d13_a1/N_d13);
                    Policy(1,curraindex,z_c,e_c,jj)=d1part;
                    Policy(3,curraindex,z_c,e_c,jj)=d3part;
                    Policy(4,curraindex,z_c,e_c,jj)=a1primepart;
                    Policy(2,curraindex,z_c,e_c,jj)=d2index_z(d3part+N_d3*(a1primepart-1));
                end
            end
        end
    end
end


end
