function [PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions)
% Note: N_d is not relevant here
% Note: simoptions.fastOLG=0: II1 will be II, everything after it will be empty
% If not using d, set l_d=0 as input

N_probs=1; % Not using N_probs
if simoptions.gridinterplayer==1
    N_probs=N_probs*2;
end
if simoptions.experienceasset==1
    N_probs=N_probs*2;
end


%% slowOLG
if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
        % Shapes:
        % V is [N_a,N_j]
        % AgentDist for basic is [N_a,N_j]
        % AgentDist for fastOLG is [N_a*N_j,1]
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            if simoptions.fastOLG==0
                II1=1:1:N_a;
                II2=ones(N_a,1);
            elseif simoptions.fastOLG==1
                II1=1:1:N_a*(N_j-1);
                II2=ones(N_a*(N_j-1),1);
                exceptlastj=[];
                exceptfirstj=[];
                justfirstj=[];
            end
        elseif simoptions.gridinterplayer==1
            if simoptions.fastOLG==0
                error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
            elseif simoptions.fastOLG==1
                II1=repelem((1:1:N_a*(N_j-1))',1,N_probs);
                II2=[];
                exceptlastj=[];
                exceptfirstj=[];
                justfirstj=[];
            end
        end
    elseif N_z>0 && N_e==0
        % Shapes:
        % V is [N_a,N_z,N_j]
        % AgentDist for basic is [N_a*N_z,N_j]
        % AgentDist for fastOLG is [N_a*N_j*N_z,1]
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            if simoptions.fastOLG==0
                II1=1:1:N_a*N_z;
                II2=ones(N_a*N_z,1);
            elseif simoptions.fastOLG==1
                II1=1:1:N_a*(N_j-1)*N_z;
                II2=ones(N_a*(N_j-1)*N_z,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
            end
        elseif simoptions.gridinterplayer==1
            if simoptions.fastOLG==0
                error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
            elseif simoptions.fastOLG==1
                II1=repelem((1:1:N_a*(N_j-1)*N_z)',1,N_probs);
                II2=[];
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
            end
        end
    elseif N_z==0 && N_e>0
        % Shapes:
        % V is [N_a,N_e,N_j]
        % AgentDist for basic is [N_a*N_e,N_j]
        % AgentDist for fastOLG is [N_a*N_j,N_e]
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            if simoptions.fastOLG==0
                II1=1:1:N_a*N_e;
                II2=ones(N_a*N_e,1);
            elseif simoptions.fastOLG==1
                II1=1:1:N_a*(N_j-1)*N_e;
                II2=ones(N_a*(N_j-1)*N_e,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
            end
        elseif simoptions.gridinterplayer==1
            if simoptions.fastOLG==0
                error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
            elseif simoptions.fastOLG==1
                II1=repelem((1:1:N_a*(N_j-1)*N_e)',1,N_probs);
                II2=[];
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
            end
        end
    elseif N_z>0 && N_e>0
        % Shapes:
        % V is [N_a,N_z,N_e,N_j]
        % AgentDist for basic is [N_a*N_z*N_e,N_j]
        % AgentDist for fastOLG is [N_a*N_j*N_z,N_e]
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            if simoptions.fastOLG==0
                II1=1:1:N_a*N_z*N_e;
                II2=ones(N_a*N_z*N_e,1);
            elseif simoptions.fastOLG==1
                II1=1:1:N_a*(N_j-1)*N_z*N_e;
                II2=ones(N_a*(N_j-1)*N_z*N_e,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
            end
        elseif simoptions.gridinterplayer==1
            if simoptions.fastOLG==0
                error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
            elseif simoptions.fastOLG==1
                II1=repelem((1:1:N_a*(N_j-1)*N_z*N_e)',1,N_probs);
                II2=[];
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
            end
        end
    end

    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            II1=1:1:N_a*(N_j-1);
            II2=ones(N_a*(N_j-1),1);
            exceptlastj=repmat((1:1:N_a)',N_j-1,1)+repelem(N_a*(0:1:N_j-2)',N_a,1); % Note: there is one use of N_j which is because we want to index AgentDist
            exceptfirstj=[];
            justfirstj=[];
        elseif simoptions.gridinterplayer==1
            II1=repelem((1:1:N_a*(N_j-1))',1,N_probs);
            II2=[];
            exceptlastj=[]; % not needed
            exceptfirstj=[];
            justfirstj=[];
        end
    elseif N_z>0 && N_e==0
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            II1=1:1:N_a*(N_j-1)*N_z;
            II2=ones(N_a*(N_j-1)*N_z,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
        elseif simoptions.gridinterplayer==1
            II1=repelem((1:1:N_a*(N_j-1)*N_z)',1,N_probs);
            II2=[];
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
        end
    elseif N_z==0 && N_e>0
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            II1=1:1:N_a*(N_j-1)*N_e;
            II2=ones(N_a*(N_j-1)*N_e,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
        elseif simoptions.gridinterplayer==1
            II1=repelem((1:1:N_a*(N_j-1)*N_e)',1,N_probs);
            II2=[];
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
        end
    elseif N_z>0 && N_e>0
        if vfoptions.gridinterplayer==0
            PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        elseif vfoptions.gridinterplayer==1
            PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if simoptions.gridinterplayer==0
            II1=1:1:N_a*(N_j-1)*N_z*N_e;
            II2=ones(N_a*(N_j-1)*N_z*N_e,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
        elseif simoptions.gridinterplayer==1
            II1=repelem((1:1:N_a*(N_j-1)*N_z*N_e)',1,N_probs);
            II2=[];
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
        end
    end
end


%% Clean up output
if simoptions.fastOLG==0
    exceptlastj=[];
    exceptfirstj=[];
    justfirstj=[];
end



end
