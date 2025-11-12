function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC1_GI_nod1_noz_e_raw(n_d2,n_a,n_semiz,n_e, N_j, d2_grid, a_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for d2,aprime and aprime2 (in GI layer)


%%
special_n_d2=ones(1,length(n_d2));
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    error('vfoptions.lowmemory=2 not supported with semi-exogenous states');
end

aind=gpuArray(0:1:N_a-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz,N_e,N_d2,'gpuArray');
% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a,N_semiz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints_jj=zeros(1,N_a,N_semiz,'gpuArray');
end

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        midpoints_Nj=zeros(N_d2,1,N_a,N_semiz,N_e,'gpuArray');

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_semiz, n_e, d2_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Treat standard problem as just being the first layer
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_Nj(:,1,level1ii,:,:)=maxindex1;
        
        % Second level based on montonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_semiz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_semiz, n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_Nj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_Nj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
            end
        end

        % Turn this into the 'midpoint'
        midpoints_Nj=max(min(midpoints_Nj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z-by-n_e
        aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d2,n_semiz,n_e,d2_gridvals,aprime_grid(aprimeindexes),a_grid,semiz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d2)+1;
        allind=d_ind+N_d2*aind+N_d2*N_a*semizind+N_d2*N_a*N_semiz*eind; % midpoint is n_d-by-1-by-n_a-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1

        midpoints_Nj=zeros(N_d2,1,N_a,N_semiz,'gpuArray');

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_semiz, special_n_e, d2_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_Nj(:,1,level1ii,:)=maxindex1;

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d2, n_semiz, special_n_e, d2_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii_e,[],2);
                    midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1); % unfortunately doesn't autofill
                end
            end

            % Turn this into the 'midpoint'
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_z
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn,n_d2,n_semiz,special_n_e,d2_gridvals,aprime_grid(aprimeindexes),a_grid,semiz_gridvals_J(:,:,N_j),e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoints_Nj(allind)),-1); % midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d2),-1); % aprimeL2ind
        end
    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*pi_e_J(1,1,:,N_j),3);    % First, switch V_Jplus1 into Kron form

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2); % effecitely nod (as just a single d2 point), so use level=2 even for first level
            entireRHS_d2ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*EV_d2;
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(entireRHS_d2ii,[],1); % no d1

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,level1ii,:,:)=maxindex1;

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex(1,2:end,:,:)-maxindex(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_semiz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,4);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1); % unfortunately doesn't autofill
                end
            end

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_semiz-by-n_e
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-n_semiz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*semizind; % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(midpoints_jj,1);

            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj); % no d1

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
        Policy(2,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
        Policy(3,:,:,:,N_j)=aprimeL2_ind; % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,4);
                entireRHS_d2iie=ReturnMatrix_d2iie+DiscountFactorParamsVec*EV_d2;
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_d2iie,[],1); % no d1

                % Just keep the 'midpoint' vesion of maxindex1 [as GI]
                midpoints_jj(1,level1ii,:)=maxindex1;

                % Second level based on montonicity
                maxgap=squeeze(max(max(maxindex(1,2:end,:)-maxindex(1,1:end-1,:),[],3)));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1); % unfortunately doesn't autofill
                    end
                end

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_semiz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*semizind; % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(midpoints_jj,1);

                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj);  % no d1
            end

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,N_j)=V_jj;
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
        Policy(2,:,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
        Policy(3,:,:,:,N_j)=aprimeL2_ind; % aprimeL2ind

    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz,n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,4);
            entireRHS_d2ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*EV_d2;
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(entireRHS_d2ii,[],1); % no d1

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,level1ii,:,:)=maxindex1;

            % Second level based on montonicity
            maxgap=squeeze(max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_semiz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
                    aprimez=aprimeindexes+N_a*semizind;
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii,:,:);
                    midpoints_jj(1,curraindex,:,:)=repelem(loweredge,1,length(curraindex),1); % unfortunately doesn't autofill
                end
            end

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a-by-n_semiz-by-n_e
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n2long-by-n_a-by-n_semiz-by-n_e
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, n_e, d2_val, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,5);
            aprimez=aprimeindexes+n2aprime*semizind; % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz,N_e]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);
            
            V_ford2_jj(:,:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,:,d2_c)=shiftdim(maxindex,1);

            midpoint_ford2_jj(:,:,:,d2_c)=squeeze(midpoints_jj);  % no d1

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
        Policy(2,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
        Policy(3,:,:,:,jj)=aprimeL2_ind; % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            d2_val=d2_gridvals(d2_c,:);
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);

            EV_d2=EV.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,4);
                entireRHS_d2iie=ReturnMatrix_d2iie+DiscountFactorParamsVec*EV_d2;
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_d2iie,[],1); % no d1

                % Just keep the 'midpoint' vesion of maxindex1 [as GI]
                midpoints_jj(1,level1ii,:,:)=maxindex1;

                % Second level based on montonicity
                maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is 1-by-1-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii))';
                        % aprime possibilities are maxgap(ii)+1-by-1-by-n_semiz
                        ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                        aprimez=aprimeindexes+N_a*semizind;
                        entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                        [~,maxindex]=max(entireRHS_ii,[],1);
                        midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(1,ii,:);
                        midpoints_jj(1,curraindex,:,:)=repelem(loweredge,length(curraindex),1); % unfortunately doesn't autofill
                    end
                end

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is 1-by-n_a-by-n_semiz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
                % aprime possibilities are n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d2, n_semiz, special_n_e, d2_val, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,5);
                aprimez=aprimeindexes+n2aprime*semizind; % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[n2long,N_a,N_semiz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,:,e_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,:,e_c,d2_c)=shiftdim(maxindex,1);

                midpoint_ford2_jj(:,:,e_c,d2_c)=squeeze(midpoints_jj); % no d1
            end

        end
        % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
        [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
        V(:,:,:,jj)=V_jj;
        Policy(1,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
        maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
        aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
        Policy(2,:,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]); % midpoint
        Policy(3,:,:,:,jj)=aprimeL2_ind; % aprimeL2ind

    end
end

%% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=adjust.*Policy(3,:,:,:,:)+(1-adjust).*(Policy(3,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:,:,:)+N_d2*(Policy(2,:,:,:,:)-1)+N_d2*N_a*(Policy(3,:,:,:,:)-1));

end
