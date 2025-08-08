function PolicyKron=KronPolicyIndexes_InfHorz(Policy, n_d, n_a, n_z, vfoptions)

% Input: Policy (l_d+l_aprime,n_a,n_z);
% Output: if no d, then Policy=zeros(N_a,N_z); % indexes the optimal choice for aprime as function of a,z 
%         if d, then Policy=zeros(2,N_a,N_z);  % indexes the optimal choice for d,aprime as function of a,z 
% This can all be a bit different based on vfoptions

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0 && isscalar(n_a) && vfoptions.gridinterplayer==0
    Policy=reshape(Policy,[N_a,N_z]);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);
end

if N_d==0 && vfoptions.gridinterplayer==0
    if isscalar(n_a)
        PolicyKron=Policy;
    elseif length(n_a)==2
        PolicyKron=Policy(1,:,:)+n_a(1)*(Policy(2,:,:)-1);
        PolicyKron=shiftdim(PolicyKron,1);
    elseif length(n_a)==3
        PolicyKron=Policy(1,:,:)+n_a(1)*(Policy(2,:,:)-1)+n_a(1)*n_a(2)*(Policy(3,:,:)-1);
        PolicyKron=shiftdim(PolicyKron,1);
    elseif length(n_a)==4
        PolicyKron=Policy(1,:,:)+n_a(1)*(Policy(2,:,:)-1)+n_a(1)*n_a(2)*(Policy(3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(4,:,:)-1);
        PolicyKron=shiftdim(PolicyKron,1);
    end
elseif N_d==0 && vfoptions.gridinterplayer==1
    if length(n_a)<=2
        PolicyKron=Policy;
    else
        PolicyKron=zeros(3,N_a,N_z,'gpuArray');
        PolicyKron(1,:,:)=Policy(1,:,:);
        if length(n_a)==3
            PolicyKron(2,:,:)=Policy(2,:,:)+n_a(2)*(Policy(3,:,:)-1);
        elseif length(n_a)==4
            PolicyKron(2,:,:)=Policy(2,:,:)+n_a(2)*(Policy(3,:,:)-1)+n_a(2)*n_a(3)*(Policy(4,:,:)-1);
        end
        PolicyKron(3,:,:)=Policy(end,:,:); % L2 index
    end

elseif N_d>0 && vfoptions.gridinterplayer==0
    PolicyKron=zeros(2,N_a,N_z,'gpuArray');
    % First, do d
    l_d=length(n_d);
    if l_d==1
        PolicyKron(1,:,:)=Policy(1,:,:);
    elseif l_d==2
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1);
    elseif l_d==3
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1)+n_d(1)*n_d(2)*(Policy(3,:,:)-1);
    elseif l_d==4
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1)+n_d(1)*n_d(2)*(Policy(3,:,:)-1)+n_d(1)*n_d(2)*n_d(3)*(Policy(4,:,:)-1);
    end
    % Then do aprime
    if isscalar(n_a)
        PolicyKron(2,:,:)=Policy(l_d+1,:,:);
    elseif length(n_a)==2
        PolicyKron(2,:,:)=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1);
    elseif length(n_a)==3
        PolicyKron(2,:,:)=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1);
    elseif length(n_a)==4
        PolicyKron(2,:,:)=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1);
    end
elseif N_d>0 && vfoptions.gridinterplayer==1
    if isscalar(n_a)
        PolicyKron=zeros(3,N_a,N_z,'gpuArray');
    else
        PolicyKron=zeros(4,N_a,N_z,'gpuArray');
    end
    % First, do d
    l_d=length(n_d);
    if l_d==1
        PolicyKron(1,:,:)=Policy(1,:,:);
    elseif l_d==2
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1);
    elseif l_d==3
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1)+n_d(1)*n_d(2)*(Policy(3,:,:)-1);
    elseif l_d==4
        PolicyKron(1,:,:)=Policy(1,:,:)+n_d(1)*(Policy(2,:,:)-1)+n_d(1)*n_d(2)*(Policy(3,:,:)-1)+n_d(1)*n_d(2)*n_d(3)*(Policy(4,:,:)-1);
    end
    % Then do aprime
    PolicyKron(2,:,:)=Policy(l_d+1,:,:);
    if ~isscalar(n_a)
        if length(n_a)==2
            PolicyKron(3,:,:)=Policy(l_d+2,:,:);
        elseif length(n_a)==3
            PolicyKron(3,:,:)=Policy(l_d+2,:,:)+n_a(2)*(Policy(l_d+3,:,:)-1);
        elseif length(n_a)==4
            PolicyKron(3,:,:)=Policy(l_d+2,:,:)+n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1);
        end
    end
    if isscalar(n_a)
        PolicyKron(3,:,:)=Policy(end,:,:); % L2 index
    else
        PolicyKron(4,:,:)=Policy(end,:,:); % L2 index
    end
end


end