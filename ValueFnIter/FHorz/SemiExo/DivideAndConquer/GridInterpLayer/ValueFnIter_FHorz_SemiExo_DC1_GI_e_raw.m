function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC1_GI_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, n_e,N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J,pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_d=[n_d1,n_d2];
n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy4=zeros(4,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% First dimension: d1, d2, aprime, aprime2

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a_grid=gpuArray(a_grid);

special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=CreateGridvals(n_d,[d1_grid; d2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
else
    eind=shiftdim((0:1:N_e-1),-2); % already includes -1
end
aind=0:1:N_a-1;
bothzind=shiftdim((0:1:N_bothz-1),-1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_semiz*N_z,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints_jj=zeros(N_d,1,N_a,N_semiz*N_z,'gpuArray');
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
if length(n_a)>1
    error('can only do gridinterplayer with one endo state (you have length(n_a)>1)')
end
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    if vfoptions.lowmemory==0

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(:,1,level1ii,:,:)=maxindex1;
        
        % Second level based on montonicity
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_bothz-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2); % ,2) as GI
                midpoints_jj(:,1,curraindex,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                midpoints_jj(:,1,curraindex,:,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
            end
        end

        % Midpoints
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);

        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, [n_d1,n_d2], n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        Policy4(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy4(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy4(3,:,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy4(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); %aprimeL2ind

    elseif vfoptions.lowmemory==1

        % for e_c=1:N_e
        %     e_val=e_gridvals_J(e_c,:,N_j);
        %     % n-Monotonicity
        %     ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
        % 
        %     % First, we want aprime conditional on (d,1,a,z,e)
        %     [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
        % 
        %     % Now, get and store the full (d,aprime)
        %     [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a,vfoptions.level1n,N_bothz]),[],1);
        % 
        %     % Store
        %     V(level1ii,:,e_c,N_j)=shiftdim(Vtempii,1);
        %     Policytemp(level1ii,:,e_c)=shiftdim(maxindex2,1); % d,aprime
        % 
        %     % Second level based on montonicity
        %     maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        %     for ii=1:(vfoptions.level1n-1)
        %         curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        %         if maxgap(ii)>0
        %             loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
        %             % loweredge is n_d-by-1-by-n_bothz
        %             aprimeindexes=loweredge+(0:1:maxgap(ii));
        %             % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
        %             ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
        %             [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
        %             V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
        %             dind=(rem(maxindex-1,N_d)+1);
        %             allind=dind+N_d*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
        %             Policytemp(curraindex,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
        %         else
        %             loweredge=maxindex1(:,1,ii,:);
        %             % Just use aprime(ii) for everything
        %             ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_bothz, special_n_e, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
        %             [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
        %             V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
        %             dind=(rem(maxindex-1,N_d)+1);
        %             allind=dind+N_d*bothzind; % loweredge is n_d-by-1-by-1-by-n_bothz
        %             Policytemp(curraindex,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1)); % loweredge(given the d and z)
        %         end
        %     end
        % 
        %     % Deal with policy for semi-exo
        %     d_ind=shiftdim(rem(Policytemp-1,N_d)+1,-1);
        %     Policy4(1,:,:,e_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        %     Policy4(2,:,:,e_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        %     Policy4(3,:,:,e_c,N_j)=shiftdim(ceil(Policytemp/N_d),-1);
        % end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz*N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,1,:,N_j),3);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12_gridvals=[d1_grid, d2_grid(d2_c)*ones(n_d1,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repmat(shiftdim(EV,-1),N_d1,1,1,1); % [d1,aprime,1,z]

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz,n_e, d12_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;
            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,level1ii,:,:)=maxindex1;

            % Second level based on montonicity
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, n_e, d12_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                    daprimez=(1:1:N_d1)'+N_d1*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,(maxgap(ii)+1),level1iidiff(ii),N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,curraindex,:,:)=loweredge;
                end
            end

        end

        % Now for the interpolation layer

        % Interpolate EV over aprime_grid
        EVinterp=interp1(a_grid,EV,aprime_grid); % CHECK RUNTIME!!!
        entireEVinterp=repelem(EVinterp,N_d1,1,1); % Note, this is only for later as it is the interpolated version
        % Note:  entireEVinterp is size [N_d1*n2aprime,1,N_semiz*N_z]

        % Midpoints
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        
        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, [n_d1,n_d2], n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
        daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_bothz,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii+entireEVinterp,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        Policy4(1,:,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy4(2,:,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy4(3,:,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy4(4,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
    %     for d2_c=1:N_d2
    %         d12_gridvals=[d1_grid, d2_grid(d2_c)*ones(n_d1,1)];
    %         % Note: By definition V_Jplus1 does not depend on d (only aprime)
    %         pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d2_c,N_j));
    % 
    %         EV=V_Jplus1.*shiftdim(pi_bothz',-1);
    %         EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    %         EV=sum(EV,2); % sum over z', leaving a singular second dimension
    % 
    %         entireEV=repelem(EV,N_d1,1,1);
    % 
    %         for e_c=1:N_e
    %             e_val=e_gridvals_J(e_c,:,N_j);
    % 
    %             % n-Monotonicity
    %             ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1);
    % 
    %             entireRHS_ii=ReturnMatrix_d2iie+DiscountFactorParamsVec*entireEV;
    % 
    %             % First, we want aprime conditional on (d,1,a,z,e)
    %             [~,maxindex1]=max(entireRHS_ii,[],2);
    % 
    %             % Now, get and store the full (d,aprime)
    %             [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
    % 
    %             % Store
    %             V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %             Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1); % d,aprime
    % 
    %             % Second level based on montonicity
    %             maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    %             for ii=1:(vfoptions.level1n-1)
    %                 curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    %                 if maxgap(ii)>0
    %                     loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
    %                     % loweredge is n_d1-by-1-by-n_bothz
    %                     aprimeindexes=loweredge+(0:1:maxgap(ii));
    %                     % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
    %                     ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
    %                     daprimez=(1:1:N_d1)'+N_d1*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
    %                     entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxgap(ii)+1),level1iidiff(ii),N_bothz]);
    %                     [Vtempii,maxindex]=max(entireRHS_ii,[],1);
    %                     V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %                     dind=(rem(maxindex-1,N_d1)+1);
    %                     allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz
    %                     Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
    %                 else
    %                     loweredge=maxindex1(:,1,ii,:);
    %                     % Just use aprime(ii) for everything
    %                     ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2);
    %                     daprimez=(1:1:N_d1)'+N_d1*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
    %                     entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,level1iidiff(ii),N_bothz]);
    %                     [Vtempii,maxindex]=max(entireRHS_ii,[],1);
    %                     V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %                     dind=(rem(maxindex-1,N_d1)+1);
    %                     allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz
    %                     Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
    %                 end
    %             end
    %         end
    %     end
    %     % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    %     [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
    %     V(:,:,:,N_j)=V_jj;
    %     Policy4(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    %     maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    %     d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    %     Policy4(1,:,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    %     Policy4(3,:,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

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
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e_J(1,1,:,jj),3);

    EVinterpfull=zeros(N_d2,n2aprime,1,N_bothz); % Note: the 1 is the N_a dimension, and N_d1 is filled out later

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12_gridvals=[d1_grid, d2_grid(d2_c)*ones(n_d1,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d2_c,jj)); % reverse order

            EV=VKronNext_j.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireEV=repmat(shiftdim(EV,-1),N_d1,1,1,1); % [d1,aprime,1,z]

            % Interpolate EV over aprime_grid
            EVinterp=interp1(a_grid,EV,aprime_grid);
            % Keep interpolated version of EV
            EVinterpfull(d2_c,:,1,:)=shiftdim(EVinterp,-1); % note, missing d1

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz,n_e, d12_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*entireEV;
            % First, we want aprime conditional on (d,1,a,z,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,level1ii,:,:)=maxindex1;

            % Second level based on montonicity
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d1-by-1-by-n_bothz-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, n_e, d12_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                    daprimez=(1:1:N_d1)'+N_d1*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,(maxgap(ii)+1),level1iidiff(ii),N_bothz,N_e]);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoints_jj((1:1:N_d1)+N_d1*(d2_c-1),1,curraindex,:,:)=loweredge;
                end
            end

        end

        % Now for the interpolation layer

        entireEVinterp=repelem(EVinterpfull,N_d1,1,1,1);

        % Midpoints
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        
        % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, [n_d1,n_d2], n_bothz, n_e, d_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
        daprimez=(1:1:N_d)'+N_d*(aprimeindexes-1)+N_d*n2aprime*shiftdim((0:1:N_bothz-1),-2); % the current aprime
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprimez(:)),[N_d*n2long,N_a,N_bothz,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind+N_d*N_a*N_bothz*eind; % midpoint is n_d-by-1-by-n_a-by-n_bothz-by-n_e
        Policy4(1,:,:,:,jj)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy4(2,:,:,:,jj)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy4(3,:,:,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy4(4,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

        
    elseif vfoptions.lowmemory==1
    %     for d2_c=1:N_d2
    %         d12_gridvals=[d1_grid, d2_grid(d2_c)*ones(n_d1,1)];
    %         % Note: By definition V_Jplus1 does not depend on d (only aprime)
    %         pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj));
    % 
    %         EV=VKronNext_j.*shiftdim(pi_bothz',-1);
    %         EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    %         EV=sum(EV,2); % sum over z', leaving a singular second dimension
    % 
    %         entireEV=repelem(EV,N_d1,1,1);
    % 
    %         for e_c=1:N_e
    %             e_val=e_gridvals_J(e_c,:,jj);
    % 
    %             % n-Monotonicity
    %             ReturnMatrix_d2iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1);
    % 
    %             entireRHS_ii=ReturnMatrix_d2iie+DiscountFactorParamsVec*entireEV;
    % 
    %             % First, we want aprime conditional on (d,1,a,z,e)
    %             [~,maxindex1]=max(entireRHS_ii,[],2);
    % 
    %             % Now, get and store the full (d,aprime)
    %             [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_bothz]),[],1);
    % 
    %             % Store
    %             V_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %             Policy_ford2_jj(level1ii,:,e_c,d2_c)=shiftdim(maxindex2,1); % d,aprime
    % 
    %             % Second level based on montonicity
    %             maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
    %             for ii=1:(vfoptions.level1n-1)
    %                 curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    %                 if maxgap(ii)>0
    %                     loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
    %                     % loweredge is n_d1-by-1-by-n_bothz
    %                     aprimeindexes=loweredge+(0:1:maxgap(ii));
    %                     % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_bothz
    %                     ReturnMatrix_iie=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
    %                     daprimez=(1:1:N_d1)'+N_d1*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
    %                     entireRHS_ii=ReturnMatrix_iie+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1*(maxgap(ii)+1),level1iidiff(ii),N_bothz]);
    %                     [Vtempii,maxindex]=max(entireRHS_ii,[],1);
    %                     V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %                     dind=(rem(maxindex-1,N_d1)+1);
    %                     allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz
    %                     Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
    %                 else
    %                     loweredge=maxindex1(:,1,ii,:);
    %                     % Just use aprime(ii) for everything
    %                     ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, special_n_d, n_bothz, special_n_e, d12_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2);
    %                     daprimez=(1:1:N_d1)'+N_d1*repelem(loweredge-1,1,1,level1iidiff(ii),1)+N_d1*N_a*shiftdim((0:1:N_bothz-1),-2); % the current aprimeii(ii):aprimeii(ii+1)
    %                     entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEV(daprimez(:)),[N_d1,level1iidiff(ii),N_bothz]);
    %                     [Vtempii,maxindex]=max(entireRHS_ii,[],1);
    %                     V_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(Vtempii,1);
    %                     dind=(rem(maxindex-1,N_d1)+1);
    %                     allind=dind+N_d1*bothzind; % loweredge is n_d1-by-1-by-1-by-n_bothz
    %                     Policy_ford2_jj(curraindex,:,e_c,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
    %                 end
    %             end
    %         end
    %     end
    %     % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    %     [V_jj,maxindex]=max(V_ford2_jj,[],4); % max over d2
    %     V(:,:,:,jj)=V_jj;
    %     Policy4(2,:,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    %     maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    %     d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    %     Policy4(1,:,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    %     Policy4(3,:,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

    end
end

%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy4(4,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy4(3,:,:,:,:)=Policy4(3,:,:,:,:)-adjust; % lower grid point
Policy4(4,:,:,:,:)=adjust.*Policy4(4,:,:,:,:)+(1-adjust).*(Policy4(4,:,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy4(1,:,:,:,:)+N_d1*(Policy4(2,:,:,:,:)-1)+N_d*(Policy4(3,:,:,:,:)-1)+N_d*N_a*(Policy4(4,:,:,:,:)-1));



end