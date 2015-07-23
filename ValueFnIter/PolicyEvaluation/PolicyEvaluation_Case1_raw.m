function VKron=PolicyEvaluation_Case1_raw(Tolerance,PolicyIndexesKron,n_d,n_a,n_z,pi_z,beta,Fmatrix)
%Evaluates the value function associated with the inputed Policy Fn
%This constitutes one of the two steps in Howard Policy Iteration approach
%to solving dynamic programming problems.

if length(n_d)==1 && n_d(1)==0
    [VKron]=PolicyEvaluation_Case1_no_d_raw(Tolerance, PolicyIndexesKron, n_a, n_z, pi_z, beta, Fmatrix);
    return
end

%disp('Policy Fn Evaluation')

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%disp('Transforming Return Fn and Initial Value Fn Matrices into Kronecker Form')
FmatrixKron=reshape(Fmatrix,[N_d*N_a,N_a,N_z]);

currdist=1;
VKron=zeros(N_a,N_z);
%Keep updating VKron using PolicyIndexesKron until VKron converges
while currdist>Tolerance
    VKronold=VKron;
    
    for a_c=1:N_a
        for z_c=1:N_z
            VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(1,a_c,z_c)+(PolicyIndexesKron(2,a_c,z_c)-1)*N_d,a_c,z_c)+beta*VKronold(PolicyIndexesKron(2,a_c,z_c),:)*pi_z(z_c,:)';
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
end

end