function SteadyStateProbDist=SteadyState_Case1(Tolerance,PolicyIndexes,PolicyIndexesInterp,n_d,n_d_cont,n_a,n_a_cont,n_z,pi_z,Nagents)
%If Nagents=0, then it will treat the agents as being on a continuum of
%weight 1.
%If Nagents is any other (integer), it will give the most likely of the
%distributions of that many agents across the various steady-states; this
%is for use with models that have a finite number of agents, rather than a
%continuum.

%Essentially the code works in two steps, the first creates
%PolicyIndexesKron (so as we can work with less dimensions)
%The second then uses this to iterate on the distribution until it
%converges to a steady state (prob) distn

N_a=prod(n_a);
N_z=prod(n_z);
number_a_vars=length(n_a);
number_a_cont_vars=length(n_a_cont);
number_d_cont_vars=length(n_d_cont);
if length(n_a_cont)==1 && n_a_cont(1)==0
    number_a_cont_vars=0;
end
if length(n_d_cont)==1 && n_d_cont(1)==0
        number_d_cont_vars=0;
end
number_cont_vars=number_d_cont_vars+number_a_cont_vars;

Interpolate=1;
if length(n_a_cont)==1 && n_a_cont(1)==0 && length(n_d_cont)==1 && n_d_cont(1)==0 %If there is no interpolation to be done
    Interpolate=0;
end

%PolicyIndexes is [number_d_vars+number_a_vars,n_a,n_s,n_z]
%We create to things here, one is PolicyIndexesKron which is used for the
%case without interpolation. The other is PolicyIndexesInterpKron which is
%used for the case with interpolation. The first contains the point to
%which to go. The second contains two points each with a probability.
if length(n_d)==1 && n_d(1)==0
    tempPolicyIndexes=reshape(PolicyIndexes,[number_a_vars,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
    PolicyIndexesKron=zeros(N_a,N_z);
    for i1=1:N_a
        for i2=1:N_z
            PolicyIndexesKron(i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(:,i1,i2));
        end
    end
else
    number_d_vars=length(n_d);
    tempPolicyIndexes=reshape(PolicyIndexes,[number_a_vars+number_d_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    PolicyIndexesKron=zeros(2,N_a,N_z);
    for i1=1:N_a
        for i2=1:N_z
            PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(1:number_d_vars,i1,i2));
            PolicyIndexesKron(2,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(number_d_vars+1:number_d_vars+number_a_vars,i1,i2));
        end
    end
end
%Now do PolicyIndexesInterpKron
if Interpolate==1
    if length(n_d)==1 && n_d(1)==0
        tempPolicyIndexesInterp=reshape(PolicyIndexesInterp,[number_a_vars+1,2^number_cont_vars,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
        PolicyIndexesInterpKron=zeros(2,2^number_cont_vars,N_a,N_z);
        for i1=1:N_a
            for i2=1:N_z
                for j=1:(2^number_cont_vars)
                    PolicyIndexesInterpKron(1,j,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexesInterp(1:number_a_vars,j,i1,i2));
                    PolicyIndexesInterpKron(2,j,i1,i2)=tempPolicyIndexesInterp(number_a_vars+1,j,i1,i2); %The probability
                end
            end
        end
    else
        number_d_vars=length(n_d);
        tempPolicyIndexesInterp=reshape(PolicyIndexesInterp,[number_a_vars+number_d_vars+1,2^number_cont_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
        PolicyIndexesInterpKron=zeros(3,2^number_cont_vars,N_a,N_z);
        for i1=1:N_a
            for i2=1:N_z
                for j=1:(2^number_cont_vars)
                    PolicyIndexesInterpKron(1,j,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexesInterp(1:number_d_vars,j,i1,i2));
                    PolicyIndexesInterpKron(2,j,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexesInterp(number_d_vars+1:number_d_vars+number_a_vars,j,i1,i2));
                    PolicyIndexesInterpKron(3,j,i1,i2)=tempPolicyIndexesInterp(number_d_vars+number_a_vars+1,j,i1,i2); %The probability
                end
            end
        end
    end
end



SteadyStateProbDistKron=ones(N_a,N_z)/(N_a*N_z);

SteadyStateProbDistKronOld=zeros(N_a,N_z);
SScurrdist=max(max(abs(SteadyStateProbDistKron-SteadyStateProbDistKronOld)));
SScounter=0;
while SScurrdist>Tolerance
    SScurrdist=sum(abs(reshape(SteadyStateProbDistKron-SteadyStateProbDistKronOld, [N_a*N_z,1])));
    SteadyStateProbDistKronOld=SteadyStateProbDistKron;
    
    SteadyStateProbDistKron=zeros(N_a,N_z);
    if Interpolate==0 %|| SScurrdist>10*Tolerance
        for a_c=1:N_a
            for z_c=1:N_z
                for zprime_c=1:N_z
                    if length(n_d)==1 && n_d(1)==0
                        optaprime=PolicyIndexesKron(a_c,z_c);
                    else
                        optaprime=PolicyIndexesKron(2,a_c,z_c);
                    end
                    SteadyStateProbDistKron(optaprime,zprime_c)=SteadyStateProbDistKron(optaprime,zprime_c)+SteadyStateProbDistKronOld(a_c,z_c)*pi_z(z_c,zprime_c);%/sum(pi_z(z_c,:));
                end
            end
        end
    else
        for a_c=1:N_a
            for z_c=1:N_z
                for j=1:(2^number_cont_vars)
                    for zprime_c=1:N_z
                        if length(n_d)==1 && n_d(1)==0
                            optaprime=PolicyIndexesInterpKron(1,j,a_c,z_c);
                            prob=PolicyIndexesInterpKron(2,j,a_c,z_c);
                        else
                            optaprime=PolicyIndexesInterpKron(2,j,a_c,z_c);
                            prob=PolicyIndexesInterpKron(3,j,a_c,z_c);
                        end
                        SteadyStateProbDistKron(optaprime,zprime_c)=SteadyStateProbDistKron(optaprime,zprime_c)+prob*SteadyStateProbDistKronOld(a_c,z_c)*pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                    end
                end
            end
        end
    end
    SScounter=SScounter+1;
    if rem(SScounter,100)==1
        SScounter
        SScurrdist
    end
end


if Nagents~=0
   SteadyStateProbDistKronTemp=reshape(SteadyStateProbDistKron, [N_a*N_z,1]);
   SteadyStateProbDistKronTemp=cumsum(SteadyStateProbDistKronTemp);
   SteadyStateProbDistKronTemp=SteadyStateProbDistKronTemp*Nagents;
   SteadyStateProbDistKronTemp=round(SteadyStateProbDistKronTemp);
   
   SteadyStateProbDistKronTemp2=SteadyStateProbDistKronTemp;
   for i=2:length(SteadyStateProbDistKronTemp)
       SteadyStateProbDistKronTemp2(i)=SteadyStateProbDistKronTemp(i)-SteadyStateProbDistKronTemp(i-1);
   end
   SteadyStateProbDistKron=reshape(SteadyStateProbDistKronTemp2,[N_a,N_z]);
end

SteadyStateProbDist=reshape(SteadyStateProbDistKron,[n_a,n_z]);

end
