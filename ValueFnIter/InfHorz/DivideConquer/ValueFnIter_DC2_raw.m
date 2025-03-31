function [V,Policy]=ValueFnIter_DC2_raw(V0, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)
% Divide-and-conquer with two endogenous states

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

d_gridvals=CreateGridvals(n_d,d_grid,1);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=[21,21];
level11ii=round(linspace(1,n_a(1),vfoptions.level1n(1)));
level11iidiff=level11ii(2:end)-level11ii(1:end-1)+1; % Note: For 2D this includes the end-points
level12jj=round(linspace(1,n_a(2),vfoptions.level1n(2)));
level12jjdiff=level12jj(2:end)-level12jj(1:end-1)+1; % Note: For 2D this includes the end-points


%% Start by setting up ReturnFn for the first-level (as we can reuse this every iteration)
ReturnMatrixLvl1=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12jj), z_gridvals, ReturnFnParamsVec, 1);
% if vfoptions.actualV0==0
%     %% Solve first-level ignoring the second level
%     % This will hopefully work to give a good initial guess for the second level
%     % We can use 'Refine' on this first level: http://discourse.vfitoolkit.com/t/pure-discretization-with-refinement/206
% 
%     % For refinement, now we solve for d*(aprime,a,z) that maximizes the ReturnFn
%     [ReturnMatrixLvl1_refined,~]=max(reshape(ReturnMatrixLvl1,[N_d,N_a1*N_a2,prod(vfoptions.level1n),N_z]),[],1);
%     ReturnMatrixLvl1_refined=shiftdim(ReturnMatrixLvl1_refined,1);
% 
%     lvl1aprimeindexes=repmat(level11ii',vfoptions.level1n(2),1)+vfoptions.level1n(1)*(repelem(level12jj',vfoptions.level1n(1),1)-1);
% 
%     % Now, do value function iteration on just this first level (we already did Refine, so no d variable)
%     [V,~]=ValueFnIter_Case1_NoD_Par2_raw(zeros(prod(vfoptions.level1n),N_z,'gpuArray'), vfoptions.level1n, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrixLvl1_refined(lvl1aprimeindexes,:,:), vfoptions.howards, vfoptions.maxhowards, 100*vfoptions.tolerance, vfoptions.maxiter);
%     % Note: uses 100*vfoptions.tolerance, as it is just an intial guess
% 
%     % Turn this V, which is currently only on the first level grid, into a full V by linear interpolation
% 
%     V=interp3(reshape(V,[vfoptions.level1n(1),vfoptions.level1n(2),N_z]),linspace(1,vfoptions.level1n(2),N_a2),linspace(1,vfoptions.level1n(1),N_a1)',(1:1:N_z));
%     % Note: For reasons known only to matlab, you use interp3(V,Xq,Yq,Zq) with X=1:n, Y=1:m, Z=1:p, where [m,n,p] = size(V).
%     %       So X and Y are 'reversed' from what you would expect them to be.
%     % Weird behaviour, presumably something to do with how Matlab views columns as the first dimension.
% 
%     if vfoptions.verbose==1
%         fprintf('Created the initial guess for V based on level 1 of divide-and-conquer \n')
%     end
% else
V=reshape(V0,[N_a1,N_a2,N_z]);
% end


%% Now that we are armed with a (hopefully decent) initial guess, we can get on with the full problem
% VKron=zeros(N_a1,N_a2,N_z,'gpuArray');
Policy=zeros(N_a1,N_a2,N_z,'gpuArray');

Ftemp=zeros(N_a1,N_a2,N_z,'gpuArray');

% Precompute
Epi_z=shiftdim(pi_z',-2); % pi_z in the form we need it to compute the expections

% For Howards we want
bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

% I want the print that tells you distance to have number of decimal points corresponding to vfoptions.tolerance
distvstolstr=['ValueFnIter: after %i iterations the dist is %4.',num2str(-round(log10(vfoptions.tolerance))),'f \n'];

currdist=Inf;
tempcounter=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter

    Vold=V;
    
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=Vold.*Epi_z;
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,3); % sum over z', leaving a singular second dimension
    % EV is (a1prime,a2prime,1,z)

    DiscountedentireEV=DiscountFactorParamsVec*reshape(repmat(shiftdim(EV,-1),N_d,1,1,1),[N_d,N_a1*N_a2,1,1,N_z]); % [d,aprime,1,1,z]

    %% Level 1
    % We can just reuse ReturnMatrixLvl1 (which has already been refined)
    entireRHS_ii=ReturnMatrixLvl1+DiscountedentireEV;

    % First, we want a1a2prime conditional on (d,1,a,z)
    [~,maxindex1]=max(entireRHS_ii,[],2);
    
    % In 2D, this all gets overwritten as each layer2-ii-jj includes the edges, so I can skip it to save time
    % % Now, get and store the full (d,aprime)
    % [~,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n(1),vfoptions.level1n(2),N_z]),[],1);
    % 
    % % Store
    % V(level11ii,level12jj,:)=shiftdim(Vtempii,1);
    % Policy(level11ii,level12jj,:)=shiftdim(maxindex2,1);

    %% Level 2
    % Split maxindex1 into a1prime and a2prime
    maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
    maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

    % Attempt for improved version
    maxgap1=squeeze(max(max(maxindex11(:,1,2:end,2:end,:)-maxindex11(:,1,1:end-1,1:end-1,:),[],5),[],1));
    maxgap2=squeeze(max(max(maxindex12(:,1,2:end,2:end,:)-maxindex12(:,1,1:end-1,1:end-1,:),[],5),[],1));
    for ii=1:(vfoptions.level1n(1)-1)
        for jj=1:(vfoptions.level1n(2)-1)
            curra1index=(level11ii(ii):1:level11ii(ii+1)); % Note: in 2D we need to include the edges
            curra2index=(level12jj(jj):1:level12jj(jj+1)); % Note: in 2D we need to include the edges

            if maxgap1(ii,jj)>0 && maxgap2(ii,jj)>0
                loweredge1=min(maxindex11(:,1,ii,jj,:),N_a1-maxgap1(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                loweredge2=min(maxindex12(:,1,ii,jj,:),N_a2-maxgap2(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-1-by-n_z
                a1primeindexes=loweredge1+repmat((0:1:maxgap1(ii,jj)),1,maxgap2(ii,jj)+1);
                a2primeindexes=loweredge2+repelem((0:1:maxgap2(ii,jj)),1,maxgap1(ii,jj)+1);
                % aprime possibilities are maxgap(ii)+1-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(a2primeindexes-1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_iijj+DiscountedentireEV(reshape(daprimez,[N_d*(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1),1,1,N_z]));  % Autofill level11iidiff(ii),level12jjdiff(jj) in the 2nd and 3rd dimensions
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=rem(maxindex-1,N_d)+1;
                a1primeind=rem(ceil(maxindex/N_d)-1,maxgap1(ii,jj)+1)+1-1; % already includes -1
                a2primeind=ceil(maxindex/(N_d*(maxgap1(ii,jj)+1)))-1; % already includes -1
                maxindexfix=dind+N_d*a1primeind+N_d*N_a1*a2primeind; % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                zind=shiftdim(0:1:N_z-1,-2); % already includes -1
                Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

                % Need to keep Ftemp for Howards policy iteration improvement
                Ftemp(curra1index,curra2index,:)=ReturnMatrix_iijj(shiftdim(maxindex,1)+N_d*(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1)*(0:1:level11iidiff(ii)-1)'+N_d*(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1)*level11iidiff(ii)*(0:1:level12jjdiff(jj)-1)+N_d*(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1)*level11iidiff(ii)*level12jjdiff(jj)*shiftdim((0:1:N_z-1),-1));

            elseif maxgap1(ii,jj)>0 && maxgap2(ii,jj)==0
                loweredge1=min(maxindex11(:,1,ii,jj,:),N_a1-maxgap1(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                loweredge2=maxindex12(:,1,ii,jj,:);
                % loweredge is 1-by-1-by-1-by-n_z
                a1primeindexes=loweredge1+(0:1:maxgap1(ii,jj));
                ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1_grid(a1primeindexes), reshape(a2_grid(loweredge2),[N_d,1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*(a1primeindexes-1)+N_d*N_a1*(loweredge2-1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_iijj+DiscountedentireEV(reshape(daprimez,[N_d*(maxgap1(ii,jj)+1)*1,1,1,N_z]));  % Autofill level11iidiff(ii),level12jjdiff(jj) in the 2nd and 3rd dimensions
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
                % no need to rework maxindex in this case
                dind=rem(maxindex-1,N_d)+1;
                %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                zind=shiftdim(0:1:N_z-1,-2); % already includes -1
                Policy(curra1index,curra2index,:)=shiftdim(maxindex+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

                % Need to keep Ftemp for Howards policy iteration improvement
                Ftemp(curra1index,curra2index,:)=ReturnMatrix_iijj(shiftdim(maxindex,1)+N_d*(maxgap1(ii,jj)+1)*(0:1:level11iidiff(ii)-1)'+N_d*(maxgap1(ii,jj)+1)*level11iidiff(ii)*(0:1:level12jjdiff(jj)-1) +N_d*(maxgap1(ii,jj)+1)*level11iidiff(ii)*level12jjdiff(jj)*shiftdim((0:1:N_z-1),-1));

            elseif maxgap1(ii,jj)==0 && maxgap2(ii,jj)>0
                loweredge1=maxindex11(:,1,ii,jj,:);
                loweredge2=min(maxindex12(:,1,ii,jj,:),N_a2-maxgap2(ii,jj)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                a2primeindexes=loweredge2+(0:1:maxgap2(ii,jj));
                ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1,N_z]), a2_grid(a2primeindexes), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(a2primeindexes-1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_iijj+DiscountedentireEV(reshape(daprimez,[N_d*(maxgap1(ii,jj)+1)*(maxgap2(ii,jj)+1),1,1,N_z]));  % Autofill level11iidiff(ii),level12jjdiff(jj) in the 2nd and 3rd dimensions
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                dind=rem(maxindex-1,N_d)+1;
                % a1primeind=1;
                % a2primeind=ceil(maxindex/N_d);
                maxindexfix=dind+N_d*N_a1*(ceil(maxindex/N_d)-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge1(dind & zind), need to 'add'  the loweredge1
                %  the a2prime is relative to loweredge2(dind & zind), need to 'add'  the loweredge2
                zind=shiftdim(0:1:N_z-1,-2); % already includes -1
                Policy(curra1index,curra2index,:)=shiftdim(maxindexfix+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

                % Need to keep Ftemp for Howards policy iteration improvement
                Ftemp(curra1index,curra2index,:)=ReturnMatrix_iijj(shiftdim(maxindex,1)+N_d*(maxgap2(ii,jj)+1)*(0:1:level11iidiff(ii)-1)'+N_d*(maxgap2(ii,jj)+1)*level11iidiff(ii)*(0:1:level12jjdiff(jj)-1) +N_d*(maxgap2(ii,jj)+1)*level11iidiff(ii)*level12jjdiff(jj)*shiftdim((0:1:N_z-1),-1));

            else % maxgap1(ii,jj)==0 && maxgap2(ii,jj)==0
                loweredge1=maxindex11(:,1,ii,jj,:);
                loweredge2=maxindex12(:,1,ii,jj,:);
                % loweredge is 1-by-1-by-1-by-n_z (a1prime&a2prime,a1,a2,z)
                ReturnMatrix_iijj=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, reshape(a1_grid(loweredge1),[N_d,1,1,1,N_z]), reshape(a2_grid(loweredge2),[N_d,1,1,1,N_z]), a1_grid(level11ii(ii):level11ii(ii+1)), a2_grid(level12jj(jj):level12jj(jj+1)), z_gridvals, ReturnFnParamsVec,2);
                daprimez=(1:1:N_d)'+N_d*(loweredge1-1)+N_d*N_a1*(loweredge2-1)+N_d*N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_iijj+DiscountedentireEV(reshape(daprimez,[N_d*1*1,1,1,N_z]));  % Autofill level11iidiff(ii),level12jjdiff(jj) in the 2nd and 3rd dimensions
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curra1index,curra2index,:)=shiftdim(Vtempii,1);
                % maxindex is anyway just the index for d, so
                dind=maxindex;
                %  the a1prime is relative to loweredge1(zind), need to 'add'  the loweredge1
                %  the a2prime is relative to loweredge2(zind), need to 'add'  the loweredge2
                zind=shiftdim(0:1:N_z-1,-2); % already includes -1
                Policy(curra1index,curra2index,:)=shiftdim(maxindex+N_d*(loweredge1(dind+N_d*zind)-1)+N_d*N_a1*(loweredge2(dind+N_d*zind)-1),1);

                % Need to keep Ftemp for Howards policy iteration improvement
                Ftemp(curra1index,curra2index,:)=ReturnMatrix_iijj(shiftdim(maxindex,1)+N_d*(0:1:level11iidiff(ii)-1)'+N_d*level11iidiff(ii)*(0:1:level12jjdiff(jj)-1) +N_d*level11iidiff(ii)*level12jjdiff(jj)*shiftdim((0:1:N_z-1),-1));

            end
        end
    end

    %% Finish up
    % Update currdist
    Vdist=V(:)-Vold(:); 
    Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards % Use Howards Policy Fn Iteration Improvement
        V=reshape(V,[N_a,N_z]);
        Ftemp=reshape(Ftemp,[N_a,N_z]);
        Policy_Vind=ceil(Policy(:)/N_d);
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=V(Policy_Vind,:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2);
            V=Ftemp+DiscountFactorParamsVec*reshape(EVKrontemp,[N_a,N_z]);
        end
        V=reshape(V,[N_a1,N_a2,N_z]);
        Ftemp=zeros(N_a1,N_a2,N_z,'gpuArray');
    end

    if vfoptions.verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist) % use enough decimal points to be able to see countdown of currdist to 0
        end
    end

    tempcounter=tempcounter+1;

end


%% Reshape output
Policyraw=reshape(Policy,[N_a1*N_a2,N_z]);
Policy=zeros(2,N_a1*N_a2,N_z,'gpuArray');
Policy(1,:,:)=rem(Policyraw-1,N_d)+1;
Policy(2,:,:)=ceil(Policyraw/N_d);


%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(V,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
else
    return
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid, a_grid);
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end



end
