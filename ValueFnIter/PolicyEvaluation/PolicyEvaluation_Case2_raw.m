function VKron=PolicyEvaluation_Case2_raw(Tolerance,PolicyIndexesKron,N_a,N_z,pi_z,Phi_aprimeKron,beta,FmatrixKron)
%Evaluates the value function associated with the inputed Policy Fn
%This constitutes one of the two steps in Howard Policy Iteration approach
%to solving dynamic programming problems.

%disp('Policy Fn Evaluation')

currdist=1;
VKron=zeros(N_a,N_z);
%Keep updating VKron using PolicyIndexesKron until VKron converges
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
        for a_c=1:N_a
            %VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c);
            %for zprime_c=1:N_z %Note: There is probably some better way to do this with matrix algebra that saves looping over zprime_c
            %(ie. more like the Howards improvement algorithm code implemented in Case 1)
            %The difficulty is just getting the indexes for VKrontemp right
            temp=0;
            for zprime_c=1:N_z
                if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                    temp=temp+VKronold(Phi_aprimeKron(PolicyIndexesKron(a_c,z_c),z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                end
            end
            VKron(a_c,z_c)=FmatrixKron(PolicyIndexesKron(a_c,z_c),a_c,z_c)+beta*temp;
        end
    end
    
    VKrondist=reshape(VKron-VKronold,[numel(VKron),1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
end

end