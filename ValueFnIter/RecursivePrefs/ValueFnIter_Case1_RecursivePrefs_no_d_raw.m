function [VKron, PolicyIndexesKron]=ValueFnIter_Case1_no_d_raw(Tolerance,VKron,N_a,N_z,pi_z,beta,FmatrixKron, Howards)
%Does pretty much exactly the same as ValueFnIter_Case1, only without any
%decision variable (n_d=0)

% N_z=prod(n_z);
% N_a=prod(n_a);

disp('Starting Value Fn Iteration')
PolicyIndexesKron=zeros(1,N_a,N_z); %first dim indexes the optimal choice for aprime rest of dimensions a,z
tempcounter=1; currdist=Inf;
%VKronold=zeros(N_a,N_z);

while currdist>Tolerance

    VKronold=VKron;

    for z_c=1:N_z
        RHSpart2=VKronold.*kron(ones(N_a,1),pi_z(z_c,:));
        RHSpart2(isnan(RHSpart2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        RHSpart2=sum(RHSpart2,2);
        for a_c=1:N_a
            %first calc the second half of the RHS (except beta)
%             RHSpart2=zeros(N_a,1); %aprime by kprime
%             for zprime_c=1:N_z
%                 if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     RHSpart2=RHSpart2+VKronold(:,zprime_c)*pi_z(z_c,zprime_c)';
%                 end
%             end
            
            entireRHS=FmatrixKron(:,a_c,z_c)+beta*RHSpart2; %aprime by 1
            
            %calculate in order, the maximizing aprime indexes
            [VKron(a_c,z_c),PolicyIndexesKron(1,a_c,z_c)]=max(entireRHS,[],1);
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement N_a*N_z
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            for z_c=1:N_z
                for a_c=1:N_a
%                     temp=VKrontemp(PolicyIndexesKron(1,a_c,z_c),:).*pi_z(z_c,:);
%                     temp(isnan(temp))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     temp=sum(temp);
                    temp=0;
                    for zprime_c=1:N_z
                        if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                            temp=temp+VKrontemp(PolicyIndexesKron(1,a_c,z_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                    VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c),a_c,z_c)+beta*temp;
                end
            end
        end
    end
    
    if rem(tempcounter,100)==0
        disp(tempcounter)
        disp(currdist)
    end

    tempcounter=tempcounter+1;
end %end of the while loop for the value fn iteration

end