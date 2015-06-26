function [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(VKron, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, beta, ReturnFn, ReturnFnParamNames, ReturnFnParams, Howards,Tolerance) %Verbose,

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

CreateCleanedReturnFn(ReturnFn, ReturnFnParamNames, ReturnFnParams);

%%
l_d=length(n_d);
l_a=length(n_a); 
l_z=length(n_z);

if l_d==1 && l_a==1 && l_z==1
    dvals=d_grid; dvals(1,1,1)=d_grid(1);
    aprimevals=shiftdim(a_grid,-1);
    avals=shiftdim(a_grid,-2);
elseif l_d==1 && l_a==1 && l_z==2
    dvals=d_grid;
    aprimevals=shiftdim(a_grid,-1);
    avals=shiftdim(a_grid,-2);
elseif l_d==1 && l_a==2 && l_z==1
    dvals=d_grid;
    a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
    a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
    a1vals=shiftdim(a_grid(1:n_a(1)),-3);
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
elseif l_d==1 && l_a==2 && l_z==2
    dvals=d_grid;
    a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
    a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
    a1vals=shiftdim(a_grid(1:n_a(1)),-3);
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
elseif l_d==2 && l_a==1 && l_z==1
    d1vals=d_grid(1:n_d(1));
    d2vals=shiftdim(d_grid(n_d(1)+n_d(1)+n_d(2)),-1);
    aprimevals=shiftdim(a_grid,-2);
    avals=shiftdim(a_grid,-3);
elseif l_d==2 && l_a==1 && l_z==2
    d1vals=d_grid(1:n_d(1));
    d2vals=shiftdim(d_grid(n_d(1)+n_d(1)+n_d(2)),-1);
    aprimevals=shiftdim(a_grid,-2);
    avals=shiftdim(a_grid,-3);
elseif l_d==2 && l_a==2 && l_z==1
    d1vals=d_grid(1:n_d(1));
    d2vals=shiftdim(d_grid(n_d(1)+n_d(1)+n_d(2)),-1);
    a1primevals=shiftdim(a_grid(1:n_a(1)),-2);
    a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
    a1vals=shiftdim(a_grid(1:n_a(1)),-4);
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-5);
elseif l_d==2 && l_a==2 && l_z==2
    d1vals=d_grid(1:n_d(1));
    d2vals=shiftdim(d_grid(n_d(1)+n_d(1)+n_d(2)),-1);
    a1primevals=shiftdim(a_grid(1:n_a(1)),-2);
    a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
    a1vals=shiftdim(a_grid(1:n_a(1)),-4);
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-5);
end

%%
z_gridvals=zeros(N_z,length(n_z),'gpuArray'); 
for i1=1:N_z
    sub=zeros(1,length(n_z));
    sub(1)=rem(i1-1,n_z(1))+1;
    for ii=2:length(n_z)-1
        sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
    end
    sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
    
    if length(n_z)>1
        sub=sub+[0,cumsum(n_z(1:end-1))];
    end
    z_gridvals(i1,:)=z_grid(sub);
end
% Somewhere in my codes I have a better way of implementing this z_gridvals when using gpu.
% But this will do for now.


%%
currdist=Inf;
while currdist>Tolerance
    VKronold=VKron;
    
    for z_c=1:N_z
        % TempReturnFn.m is created by CreateCleanedReturnFn() as part of ValueFnIter_Case1()
        if l_d==1 && l_a==1 && l_z==1
            zvals=z_gridvals(z_c);
            ReturnMatrix_z=arrayfun(@TempReturnFn, dvals, aprimevals, avals, zvals);
        elseif l_d==1 && l_a==1 && l_z==2
            z1vals=z_gridvals(z_c,1);
            z2vals=z_gridvals(z_c,1);
            ReturnMatrix_z=arrayfun(@TempReturnFn, dvals, aprimevals, avals, z1vals,z2vals);
        elseif l_d==1 && l_a==2 && l_z==1
            zvals=z_gridvals(z_c);
            ReturnMatrix_z=arrayfun(@TempReturnFn, dvals, a1primevals,a2primevals, a1vals,a2vals, zvals);
        elseif l_d==1 && l_a==2 && l_z==2
            z1vals=z_gridvals(z_c,1);
            z2vals=z_gridvals(z_c,1);
            ReturnMatrix_z=arrayfun(@TempReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals);
        elseif l_d==2 && l_a==1 && l_z==1
            zvals=z_gridvals(z_c);
            ReturnMatrix_z=arrayfun(@TempReturnFn, d1vals,d2vals, aprimevals, avals, zvals);
        elseif l_d==2 && l_a==1 && l_z==2
            z1vals=z_gridvals(z_c,1);
            z2vals=z_gridvals(z_c,1);
            ReturnMatrix_z=arrayfun(@TempReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals);
        elseif l_d==2 && l_a==2 && l_z==1
            zvals=z_gridvals(z_c);
            ReturnMatrix_z=arrayfun(@TempReturnFn, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, zvals);
        elseif l_d==2 && l_a==2 && l_z==2
            z1vals=z_gridvals(z_c,1);
            z2vals=z_gridvals(z_c,1);
            ReturnMatrix_z=arrayfun(@TempReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals);
        end
        
        ReturnMatrix_z=reshape(ReturnMatrix_z,[N_d*N_a,N_a]);
        
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=VKronold.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;
             
        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
        Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex); 
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    if isfinite(currdist) %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            VKrontemp=VKron;
            
            EVKrontemp=VKrontemp(ceil(PolicyIndexes/N_d),:);
            EVKrontemp=EVKrontemp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+beta*EVKrontemp;
        end
    end
    
%     if Verbose==1
%         if rem(tempcounter,100)==0
%             disp(tempcounter)
%             disp(currdist)
%         end
%         tempcounter=tempcounter+1;
%     end
    
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end