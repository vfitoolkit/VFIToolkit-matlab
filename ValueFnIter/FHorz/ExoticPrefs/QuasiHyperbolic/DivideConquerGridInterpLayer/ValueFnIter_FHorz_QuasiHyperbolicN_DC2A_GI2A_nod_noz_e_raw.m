function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicN_DC2A_GI2A_nod_noz_e_raw(n_a, n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC2A_GI2A_nod_noz_e_raw.
% divide-and-conquer in the first endo state (a2 enumerated in full), plus grid interpolation layer.
% No d. No z. Has e (iid, e-expectation via pi_e_J). GPU (parallel==2 only).
% lowmemory: =0 vectorize over e, =1 loop over e
%
% Naive: Valt_j   = max u + beta*E[V_{j+1}]         (used as EVsource)
%        Vtilde_j = max u + beta_0*beta*E[V_{j+1}]  (agent's choice)

N_a=prod(n_a);
N_e=prod(n_e);

Valt=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % first dim is (a1prime midpoint,a2prime,a1prime L2)
PolicyL2flag=2*ones(1,N_a,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper
Policyalt=zeros(3,N_a,N_e,N_j,'gpuArray'); % exponential discounter optimal choice
PolicyL2flagalt=2*ones(1,N_a,N_e,N_j,'gpuArray');

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);
% aprime_grid=[a1prime_grid; a2_grid];

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a12ind=repmat(gpuArray(0:1:N_a1-1),1,N_a2)+N_a1*repelem(gpuArray(0:1:N_a2-1),1,N_a1);
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,N_e,'gpuArray');
    eind=shiftdim(gpuArray(0:1:N_e-1),-1); % already includes -1
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a2,N_a1,N_a2,'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        %Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii_dc,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_L2,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*eind); % a1prime midpoint
        Policy(2,:,:,N_j)=maxindexL2a2; % a2prime
        Policy(3,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_val, ReturnFnParamsVec,1);

            %Calc the max and it's index
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii_e_dc,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e,a1prime_grid(a1primeindexes),a2_grid,a1_grid,a2_grid, e_val, ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_L2_e,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            Valt(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(3,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_L2_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2_e(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end

    Vtilde=Valt;
    % terminal period: QH and exponential discounter coincide
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j);
    PolicyL2flagalt(1,:,:,N_j)=PolicyL2flag(1,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    Vtilde=zeros(N_a,N_e,N_j,'gpuArray');

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j+1),2); % Using V_Jplus1

    DiscountedEV=beta*reshape(EV,[N_a1,N_a2,1,1]);  % autoexpand (a)
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_a1,N_a2,1,1]);  % autoexpand (a)
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEVinterp_tilde=interp1(a1_grid,DiscountedEV_tilde,a1prime_grid);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                aprime=aprimeindexes+N_a1*a2ind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1),N_a2,1,N_a2,N_e]));  % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);
        Valt(:,:,N_j)=shiftdim(Vtempii,1);
        Policyalt(1,:,:,N_j)=midpoints_jj(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*eind); % a1prime midpoint
        Policyalt(2,:,:,N_j)=maxindexL2a2alt; % a2prime
        Policyalt(3,:,:,N_j)=maxindexL2a1alt; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_loweralt = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        linidx_upperalt = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        PolicyL2flagalt(1,:,:,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                aprime=aprimeindexes+N_a1*a2ind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_e]));  % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        Vtilde(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*eind); % a1prime midpoint
        Policy(2,:,:,N_j)=maxindexL2a2; % a2prime
        Policy(3,:,:,N_j)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_val, ReturnFnParamsVec,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1),N_a2,1,N_a2]));  % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_L2=ReturnMatrix_L2_e+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);
            Valt(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policyalt(1,:,e_c,N_j)=midpoints_jj(maxindexL2a2alt+N_a2*a12ind); % a1prime midpoint
            Policyalt(2,:,e_c,N_j)=maxindexL2a2alt; % a2prime
            Policyalt(3,:,e_c,N_j)=maxindexL2a1alt; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_loweralt = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt = (ReturnMatrix_L2_e(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2_e(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            PolicyL2flagalt(1,:,e_c,N_j) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV_tilde;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV_tilde(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2]));  % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_L2=ReturnMatrix_L2_e+reshape(DiscountedEVinterp_tilde(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            Vtilde(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,e_c,N_j)=maxindexL2a2; % a2prime
            Policy(3,:,e_c,N_j)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_L2_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2_e(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,e_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EV=sum(Valt(:,:,jj+1).*pi_e_J(1,:,jj+1),2); % naive: continuation is the exponential value fn

    DiscountedEV=beta*reshape(EV,[N_a1,N_a2,1,1]);  % autoexpand (a)
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_a1,N_a2,1,1]);  % autoexpand (a)
    % Interpolate EV over aprime_grid
    DiscountedEVinterp=interp1(a1_grid,DiscountedEV,a1prime_grid);
    DiscountedEVinterp_tilde=interp1(a1_grid,DiscountedEV_tilde,a1prime_grid);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% Valt (beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                aprime=aprimeindexes+N_a1*a2ind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1),N_a2,1,N_a2,N_e])); % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
        maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
        maxindexL2a2alt=ceil(maxindexL2alt/n2long);
        Valt(:,:,jj)=shiftdim(Vtempii,1);
        Policyalt(1,:,:,jj)=midpoints_jj(maxindexL2a2alt+N_a2*a12ind+N_a2*N_a*eind); % a1prime midpoint
        Policyalt(2,:,:,jj)=maxindexL2a2alt; % a2prime
        Policyalt(3,:,:,jj)=maxindexL2a1alt; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_loweralt = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        linidx_upperalt = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        isInfLoweralt = (ReturnMatrix_L2(linidx_loweralt) == -Inf);
        isInfUpperalt = (ReturnMatrix_L2(linidx_upperalt) == -Inf);
        inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
        inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
        PolicyL2flagalt(1,:,:,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde; % autofill e

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_jj(1,:,level1ii,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                aprime=aprimeindexes+N_a1*a2ind;
                entireRHS_ii=ReturnMatrix_ii_dc+DiscountedEV_tilde(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2,N_e])); % autoexpand level1iidiff(ii) in 3rd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,:,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                midpoints_jj(1,:,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a2-by-n_a1-by-n_a2-by-n_e
        a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2-by-n_e
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn,n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        aprime=a1primeindexes+N_a1fine*a2ind;
        entireRHS_L2=ReturnMatrix_L2+reshape(DiscountedEVinterp_tilde(aprime),[n2long*N_a2,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        maxindexL2a1=rem(maxindexL2-1,n2long)+1;
        maxindexL2a2=ceil(maxindexL2/n2long);
        Vtilde(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind+N_a2*N_a*eind); % a1prime midpoint
        Policy(2,:,:,jj)=maxindexL2a2; % a2prime
        Policy(3,:,:,jj)=maxindexL2a1; % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind + n2long*N_a2*N_a*eind;
        isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        PolicyL2flag(1,:,:,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, e_val, ReturnFnParamsVec,1);

            %% Valt (beta)
            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV; % autofill e

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap_V=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap_V(ii))';
                    % aprime possibilities are (maxgap_V(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV(reshape(aprime,[(maxgap_V(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_L2=ReturnMatrix_L2_e+reshape(DiscountedEVinterp(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2alt]=max(entireRHS_L2,[],1);
            maxindexL2a1alt=rem(maxindexL2alt-1,n2long)+1;
            maxindexL2a2alt=ceil(maxindexL2alt/n2long);
            Valt(:,e_c,jj)=shiftdim(Vtempii,1);
            Policyalt(1,:,e_c,jj)=midpoints_jj(maxindexL2a2alt+N_a2*a12ind); % a1prime midpoint
            Policyalt(2,:,e_c,jj)=maxindexL2a2alt; % a2prime
            Policyalt(3,:,e_c,jj)=maxindexL2a1alt; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_loweralt = 1      + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            linidx_upperalt = n2long + n2long*(maxindexL2a2alt-1) + n2long*N_a2*a12ind;
            isInfLoweralt = (ReturnMatrix_L2_e(linidx_loweralt) == -Inf);
            isInfUpperalt = (ReturnMatrix_L2_e(linidx_upperalt) == -Inf);
            inLowerStrictalt = (maxindexL2a1alt >= 2)         & (maxindexL2a1alt <= n2short+1);
            inUpperStrictalt = (maxindexL2a1alt >= n2short+3) & (maxindexL2a1alt <= n2long-1);
            PolicyL2flagalt(1,:,e_c,jj) = 2 + (inLowerStrictalt & isInfLoweralt) - (inUpperStrictalt & isInfUpperalt);
            %% Vtilde (beta0*beta)
            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV_tilde; % autofill e

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(1,:,level1ii,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, e_val, ReturnFnParamsVec,3);
                    aprime=aprimeindexes+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_e_dc+DiscountedEV_tilde(reshape(aprime,[(maxgap(ii)+1),N_a2,1,N_a2])); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,:,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    midpoints_jj(1,:,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a1-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a2-by-n_a1-by-n_a2
            a1primeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a2-by-n_a1-by-n_a2
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_e,a1prime_grid(a1primeindexes),a2_grid, a1_grid, a2_grid,e_val, ReturnFnParamsVec,2);
            aprime=a1primeindexes+N_a1fine*a2ind;
            entireRHS_L2=ReturnMatrix_L2_e+reshape(DiscountedEVinterp_tilde(aprime),[n2long*N_a2,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            maxindexL2a1=rem(maxindexL2-1,n2long)+1;
            maxindexL2a2=ceil(maxindexL2/n2long);
            Vtilde(:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,jj)=midpoints_jj(maxindexL2a2+N_a2*a12ind); % a1prime midpoint
            Policy(2,:,e_c,jj)=maxindexL2a2; % a2prime
            Policy(3,:,e_c,jj)=maxindexL2a1; % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            linidx_lower = 1      + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            linidx_upper = n2long + n2long*(maxindexL2a2-1) + n2long*N_a2*a12ind;
            isInfLower = (ReturnMatrix_L2_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2_e(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            PolicyL2flag(1,:,e_c,jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        end
    end
end




%% Currently Policy(1,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=[Policy; PolicyL2flag];

adjustalt=(Policyalt(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policyalt(1,:,:,:)=Policyalt(1,:,:,:)-adjustalt; % lower grid point
Policyalt(3,:,:,:)=adjustalt.*Policyalt(3,:,:,:)+(1-adjustalt).*(Policyalt(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policyalt=[Policyalt; PolicyL2flagalt];



end
