function [PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_semiz,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions)
% Note: N_d is not relevant here
% Note: simoptions.fastOLG=0: II1 will be II, everything after it will be empty
% If not using d, set l_d=0 as input

N_probs=1; % Not using N_probs
if simoptions.gridinterplayer==1
    N_probs=N_probs*2;
end
if simoptions.experienceasset>=1 || simoptions.experienceassetz>=1
    N_probs=N_probs*2;
end

% Flag-aware GI workers propagate an extra PolicyL2flag channel through UnKron;
% under gridinterplayer==1 the extra slot is always present.
flag_extra=(vfoptions.gridinterplayer==1);

%% Semi-exogenous states: PolicyIndexesPath carries the composite bothz=(semiz,z) dimension.
% The per-tt index sets (II1/exceptlastj/...) are not used; the SemiExo dist raws build their own internally.
if N_semiz>0
    N_bothz=N_semiz*max(N_z,1); % composite (semiz,z), in place of N_z
    nrows=l_d+l_aprime+2*flag_extra; % d1,d2,aprime rows (+ L2index,L2flag for grid interpolation)
    % Same ordering convention as the generic: slowOLG is (a,bothz,[e],j); fastOLG is (a,j,bothz,[e])
    if transpathoptions.fastOLG==0
        if N_e==0
            PolicyIndexesPath=zeros(nrows,N_a,N_bothz,N_j,T-1,'gpuArray'); % Periods 1 to T-1
        else
            PolicyIndexesPath=zeros(nrows,N_a,N_bothz,N_e,N_j,T-1,'gpuArray'); % Periods 1 to T-1
        end
    else % fastOLG
        if N_e==0
            PolicyIndexesPath=zeros(nrows,N_a,N_j,N_bothz,T-1,'gpuArray'); % Periods 1 to T-1
        else
            PolicyIndexesPath=zeros(nrows,N_a,N_j,N_bothz,N_e,T-1,'gpuArray'); % Periods 1 to T-1
        end
    end
    II1=[]; II2=[]; exceptlastj=[]; exceptfirstj=[]; justfirstj=[];
    return
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
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
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
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
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
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
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
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
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
            II1=1:1:N_a*(N_j-1);
            II2=ones(N_a*(N_j-1),1);
            exceptlastj=repmat((1:1:N_a)',N_j-1,1)+repelem(N_a*(0:1:N_j-2)',N_a,1); % Note: there is one use of N_j which is because we want to index AgentDist
            exceptfirstj=[];
            justfirstj=[];
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_j,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
            II1=1:1:N_a*(N_j-1)*N_z;
            II2=ones(N_a*(N_j-1)*N_z,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_j,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
            II1=1:1:N_a*(N_j-1)*N_e;
            II2=ones(N_a*(N_j-1)*N_e,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
        else
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
            PolicyIndexesPath=zeros(l_d+l_aprime+1+flag_extra,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
        end
        if N_probs==1
            II1=1:1:N_a*(N_j-1)*N_z*N_e;
            II2=ones(N_a*(N_j-1)*N_z*N_e,1);
            exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
            justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
        else
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
