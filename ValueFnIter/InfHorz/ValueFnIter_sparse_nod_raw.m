function [VKron, Policy]=ValueFnIter_sparse_nod_raw(VKron,N_a,N_z,pi_z,DiscountFactorParamsVec,ReturnMatrix,Howards,MaxHowards,Tolerance,maxiter)
%Does pretty much exactly the same as ValueFnIter_Case1, only without any 
% decision variable (n_d=0)

pi_z_transpose = transpose(pi_z);
pi_z_alt       = shiftdim(pi_z_transpose,-1);

% Precompute variables for Howard improvement step
NaVec = gpuArray.colon(1,N_a)';
NzVec = gpuArray.colon(1,N_z)';
a_ind = repmat(NaVec,N_z,1);
z_ind = repelem(NzVec,N_a,1);
ind   = a_ind+(z_ind-1)*N_a;

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=VKronold.*pi_z_alt;
    EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

    %Calc the max and it's index
    [VKron,Policy]=max(entireRHS,[],1);

    VKron=shiftdim(VKron,1); % a by z

    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    %disp(currdist)

    % Use Howards Policy Fn Iteration Improvement 
    % (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<MaxHowards

        % Reshape Policy as column vector with size [N_a*N_z,1] 
        Policy_vec=reshape(Policy,[N_a*N_z,1]); 
        % Linear indices of the elements you want in ReturnMatrix
        linIdx = Policy_vec + (a_ind-1)*N_a + (z_ind-1)*N_a*N_a;
        Ftemp = ReturnMatrix(linIdx);
        % Build large sparse matrix
        indp  = Policy_vec+(z_ind-1)*N_a;
        Qmat  = sparse(ind,indp,1,N_a*N_z,N_a*N_z);
        % Howard iterations, with Ftemp and Qmat precomputed
        for Howards_counter=1:Howards
            EV_howard = VKron*pi_z_transpose; % (a',z)
            EV_howard = reshape(EV_howard,[N_a*N_z,1]);
            VKron = Ftemp+DiscountFactorParamsVec*Qmat*EV_howard;
            VKron = reshape(VKron,[N_a,N_z]);
        end
    end
    tempcounter=tempcounter+1;
end %end while loop

Policy=reshape(Policy,[N_a,N_z]);

end %end function "ValueFnIter_nod_raw_sparse"
