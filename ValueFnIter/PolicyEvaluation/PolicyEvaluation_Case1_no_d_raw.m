function VKron=PolicyEvaluation_Case1_no_d_raw(Tolerance,PolicyIndexesKron,n_a,n_z,pi_z,beta,Fmatrix)
%Evaluates the value function associated with the inputed Policy Fn
%This constitutes one of the two steps in Howard Policy Iteration approach
%to solving dynamic programming problems.

%This code contains three possible approaches to this problem. The first
%based on matrix inversion, and the later two on iteration (the second uses
%matrix multiplication and the third for loops). The first two are left
%commented out on the grounds that in the experience of the author the
%third approach appears to run fastest.

%disp('Policy Fn Evaluation')

N_a=prod(n_a);
N_z=prod(n_z);

% if length(n_z)==1 && n_z(1)==0;
%     num_z_vars=0;
%     N_z=1;
% else
%     num_z_vars=length(n_z);
% end

%disp('Transforming Return Fn and Initial Value Fn Matrices into Kronecker Form')
FmatrixKron=reshape(Fmatrix,[N_a,N_a,N_z]);

% disp('Method1')
% %We want to calc the value function corresponding to the Policy Fn, we will
% %do this by matrix arithmetic. Note that V=F(g)+beta*PV can be rearranged
% %to yield V=(1-beta*P)^(-1) F(g). It is this later formula that we will evaluate.
% %First lets calculate P=gQ.
% P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
% for a_c=1:N_a
%     for z_c=1:N_z
%         for zprime_c=1:N_z
%             optaprime=PolicyIndexesKron(1,a_c,z_c);
%             P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%         end
%     end
% end
% P=reshape(P,[N_a*N_z,N_a*N_z]);
% %Now calculate F(g)
% F_g=zeros(N_a,N_z);
% for a_c=1:N_a
%     for z_c=1:N_z
%         F_g(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c),a_c,z_c);
%     end
% end
% F_g=reshape(F_g,[N_a*N_z,1]);
% %Now evaluate the formula
% VKron=((eye(N_a*N_z)-beta*P)^(-1))*F_g;
% %And reshape the solution back into the outputted form
% VKron=reshape(VKron,[N_a,N_z]);
% 
% disp('Method2')
% %The alternative to matrix inversion approach would be to calculate it
% %iteratively. This is done here matrix multiplication.
% %First lets calculate P=gQ.
% P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
% for a_c=1:N_a
%     for z_c=1:N_z
%         for zprime_c=1:N_z
%             optaprime=PolicyIndexesKron(1,a_c,z_c);
%             P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%         end
%     end
% end
% P=reshape(P,[N_a*N_z,N_a*N_z]);
% %Now calculate F(g)
% F_g=zeros(N_a,N_z);
% for a_c=1:N_a
%     for z_c=1:N_z
%         F_g(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c),a_c,z_c);
%     end
% end
% F_g=reshape(F_g,[N_a*N_z,1]);
% currdist=1;
% VKron=zeros(N_a*N_z,1);
% while currdist>(Tolerance*10);
%     VKronold=VKron;
%     VKron=F_g+beta*P*VKronold;
%     
%     VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
%     currdist=max(abs(VKrondist));
% end
% toc

%disp('Method3')
%The alternative to matrix inversion approach would be to calculate it
%iteratively. This is done here using for loops.
currdist=1;
VKron=zeros(N_a,N_z);
%Keep updating VKron using PolicyIndexesKron until VKron converges
while currdist>(Tolerance*10);
    VKronold=VKron;
    
    for a_c=1:N_a
        for z_c=1:N_z
            temp=0;
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    temp=temp+VKronold(PolicyIndexesKron(1,a_c,z_c),zprime_c)*pi_z(z_c,zprime_c);
                end
            end
            VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c),a_c,z_c)+beta*temp;
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
end

end