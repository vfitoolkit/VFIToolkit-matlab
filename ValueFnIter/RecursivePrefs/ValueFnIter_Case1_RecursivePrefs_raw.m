function [VKron,PolicyIndexesKron]=ValueFnIter_Case1_raw(Tolerance, VKron, N_d,N_a,N_z, pi_z, beta, FmatrixKron, Howards)
%Does exactly the same as ValueFnIter_Case1, but does not reshape input and
%output (so these must already be in kron form) and only returns the policy
%function (no value fn). It also does not bother to check sizes.

if N_d==0
    [VKron,PolicyIndexesKron]=ValueFnIter_Case1_no_d_raw(Tolerance, VKron, N_a, N_z, pi_z, beta, FmatrixKron, Howards);
    return
end

disp('Starting Value Fn Iteration')

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

PolicyIndexesKron=zeros(2,N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
tempcounter=1; currdist=Inf;

while currdist>Tolerance
    
    VKronold=VKron;

    for z_c=1:N_z
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        RHSpart2=zeros(N_a,1); %aprime
        for zprime_c=1:N_z
            if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                RHSpart2=RHSpart2+VKronold(:,zprime_c)*pi_z(z_c,zprime_c);
            end
        end
        RHSpart2full=kron(RHSpart2,ones(N_d,1));

        for a_c=1:N_a
            %Calc the RHS
            entireRHS=FmatrixKron(:,a_c,z_c)+beta*RHSpart2full; %d by aprime by 1
            
            %Calc the max and it's index
            [VKron(a_c,z_c),maxindex]=max(entireRHS);
            PolicyIndexesKron(:,a_c,z_c)=[rem(maxindex,N_d);floor(maxindex/N_d)+1]; %[d;aprime]
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    
    if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z                
                for a_c=1:N_a
%                     temp=0;
%                     for zprime_c=1:N_z
%                         if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             temp=temp+VKrontemp(PolicyIndexesKron(2,a_c,z_c),zprime_c)*pi_z(z_c,zprime_c);
%                         end
%                     end

                    VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c)+(PolicyIndexesKron(2,a_c,z_c)-1)*N_d,a_c,z_c)+beta*VKrontemp(PolicyIndexesKron(2,a_c,z_c),:)*pi_z(z_c,:)';
                end
            end
        end
    end
    
    if rem(tempcounter,100)==0
        disp(tempcounter)
        disp(currdist)
    end

    tempcounter=tempcounter+1;
end

%disp('Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form')
%V=reshape(VKron,[n_a,n_z]);
%PolicyIndexes=UnKronPolicyIndexes_Case1(PolicyIndexesKron, n_d, n_a, n_z);

end