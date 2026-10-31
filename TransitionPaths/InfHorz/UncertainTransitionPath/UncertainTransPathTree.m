function [Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions)
% The tree of histories of an uncertain transition path, and the price and parameter paths put onto it.
% Shared setup of UncertainTransitionPath_InfHorz and of the commands that evaluate things on its tree
% (ValueFnOnUncertainTransPath_InfHorz, AgentDistOnUncertainTransPath_InfHorz, EvalFnOnUncertainTransPath_AggVars_InfHorz),
% so that they all number the nodes the same way. Also callable directly, to get the tree:
%    Tree=UncertainTransPathTree(T,T1,T2,pi_S,S_initial)
%
% THE AGGREGATE SHOCK S. Takes values 1,...,N_S, N_S=size(pi_S,1).
%   T1,...,T2 are the periods in which agents FACE uncertainty: in each period t=T1,...,T2 next period's S is
%     unknown, and S_{t+1} is drawn from pi_S(S_t,:). pi_S is N_S-by-N_S, or N_S-by-N_S-by-(T2-T1+1) in which case
%     pi_S(:,:,k) is the transition from period T1+k-1 to period T1+k.
%   So S_t=S_initial in periods 1,...,T1 (period 1 is always S_initial), the draws land in periods T1+1,...,T2+1, and
%   S is frozen from period T2+1 on: S_t=S_{T2+1} for every t>T2+1. Example: a policy that may be repealed in any of
%   periods 2,...,80 is T1=1, T2=79.
% T is the final period of every history: every node in period T is at the final stationary eqm of its S.
%
% THE TREE. A node is a history (S_1,...,S_t) of positive probability; zero entries of pi_S prune the tree, so with an
% absorbing state the number of histories grows linearly rather than as N_S^(T2-T1+1). Nodes are numbered DATE MAJOR:
% all the period 1 nodes, then all the period 2 nodes, and so on; the children of a node are numbered consecutively,
% in increasing S. So a tree with a single history numbers its nodes 1,...,T, the same as a deterministic path.
%   Tree.nodedate, .nodeS, .nodeparent (0 for a period 1 node), .nodeprob (unconditional), .nodecondprob (given the parent),
%   .nodefirstchild, .nodenchild
%   Tree.histories (nHist-by-T, the nodes of each history), .historyprob, .historyS (nHist-by-T)
% A SEGMENT is a maximal chain of nodes each with a single child. It is an ordinary deterministic transition, so the
% commands that evaluate things on the tree do it one segment at a time, with the deterministic commands. A segment
% ends at a node with several children (a branching node) or in period T.
%   Tree.segnodes{ss} (the nodes of segment ss, in date order), .segparent (the parent node of its first node, 0 if
%   none), .nodeseg (the segment each node is in). Segments are numbered by the date of their first node, so a child
%   segment always has a larger number than its parent.
%
% PricePath and ParamPath (optional) are put onto the tree, giving PricePathNodes.(name) and ParamPathNodes.(name), each
% 1-by-nNodes. Each entry of PricePath can be
%   1-by-nNodes   a value for each node (this is how UncertainTransitionPath_InfHorz returns prices)
%   N_S-by-T      the value in period t if S_t=s, the same for every history with S_t=s
%   1-by-T        the value in period t, the same for every history
% Each entry of ParamPath can be N_S-by-T or 1-by-T (as for PricePath, either orientation). A parameter that depends on
% S is N_S-by-T. Prices and parameters must be scalars.

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~isfield(transpathoptions,'maxnhistories')
    transpathoptions.maxnhistories=1000;
end

%% Check the inputs
N_S=size(pi_S,1);
if size(pi_S,2)~=N_S
    error('UncertainTransPathTree: pi_S must be N_S-by-N_S, or N_S-by-N_S-by-(T2-T1+1)')
end
if ~(T1>=1 && T2>=T1 && T2<=T-2 && T1==round(T1) && T2==round(T2) && T==round(T))
    error('UncertainTransPathTree: need integers with 1<=T1<=T2<=T-2 (T1 to T2 are the periods in which agents face uncertainty about next period; the last draw, in period T2+1, must come before the final period T)')
end
nWindow=T2-T1+1;
if size(pi_S,3)~=1 && size(pi_S,3)~=nWindow
    error('UncertainTransPathTree: pi_S must be N_S-by-N_S, or N_S-by-N_S-by-(T2-T1+1); it is N_S-by-N_S-by-%i with T2-T1+1=%i',size(pi_S,3),nWindow)
end
if size(pi_S,3)==1
    pi_S=repmat(pi_S,1,1,nWindow);
end
pi_S=gather(pi_S);
if any(pi_S(:)<0)
    error('UncertainTransPathTree: pi_S has negative entries')
end
if max(abs(sum(pi_S,2)-1),[],'all')>10^(-9)
    error('UncertainTransPathTree: every row of pi_S must sum to one (the largest error is %.3e)',max(abs(sum(pi_S,2)-1),[],'all'))
end
if ~(isscalar(S_initial) && S_initial==round(S_initial) && S_initial>=1 && S_initial<=N_S)
    error('UncertainTransPathTree: S_initial must be one of 1,...,N_S')
end

%% Count the histories before building anything, so a tree that is too big is refused before it is allocated
countS=zeros(N_S,1); % the number of positive probability histories ending in each S
countS(S_initial)=1;
nPrefix=zeros(T,1); % the number of nodes in each period
for tt=1:T
    if tt>=T1+1 && tt<=T2+1 % S_tt is drawn, from the S of period tt-1 (one of T1,...,T2)
        countS=((pi_S(:,:,tt-T1)>0)')*countS;
    end
    nPrefix(tt)=sum(countS);
end
nHist=nPrefix(T);
if nHist>transpathoptions.maxnhistories
    error('UncertainTransPathTree: the tree has %i histories, more than transpathoptions.maxnhistories=%i (and %i nodes, so every value fn and agent dist is stored %i times). Raise transpathoptions.maxnhistories if this is intended.',nHist,transpathoptions.maxnhistories,sum(nPrefix),sum(nPrefix))
end
nNodes=sum(nPrefix);

%% Build the nodes, period by period
nodedate=zeros(nNodes,1);
nodeS=zeros(nNodes,1);
nodeparent=zeros(nNodes,1);
nodeprob=zeros(nNodes,1);
nodecondprob=zeros(nNodes,1);
nodefirstchild=zeros(nNodes,1);
nodenchild=zeros(nNodes,1);
% The period 0 'node' is not stored: it is S_initial with probability one
frontierS=S_initial;
frontierprob=1;
frontiernode=0;
cc=0; % nodes so far
for tt=1:T
    firstnew=cc+1;
    for ff=1:length(frontiernode)
        if tt>=T1+1 && tt<=T2+1 % S_tt is drawn, from the S of period tt-1 (one of T1,...,T2)
            childS=find(pi_S(frontierS(ff),:,tt-T1)>0);
            childcondprob=pi_S(frontierS(ff),childS,tt-T1);
        else
            childS=frontierS(ff);
            childcondprob=1;
        end
        nc=length(childS);
        nodedate(cc+1:cc+nc)=tt;
        nodeS(cc+1:cc+nc)=childS;
        nodeparent(cc+1:cc+nc)=frontiernode(ff);
        nodecondprob(cc+1:cc+nc)=childcondprob;
        nodeprob(cc+1:cc+nc)=frontierprob(ff)*childcondprob;
        if frontiernode(ff)>0
            nodefirstchild(frontiernode(ff))=cc+1;
            nodenchild(frontiernode(ff))=nc;
        end
        cc=cc+nc;
    end
    frontiernode=(firstnew:cc)';
    frontierS=nodeS(frontiernode);
    frontierprob=nodeprob(frontiernode);
end

%% The histories: each node in period T, followed back to period 1
histories=zeros(nHist,T);
histories(:,T)=frontiernode;
for tt=T-1:-1:1
    histories(:,tt)=nodeparent(histories(:,tt+1));
end
historyprob=nodeprob(frontiernode);
historyS=reshape(nodeS(histories),[nHist,T]);

%% The segments
% A node starts a segment if it has no parent or its parent has several children
isstart=(nodeparent==0);
isstart(nodeparent>0)=(nodenchild(nodeparent(nodeparent>0))>1);
segstart=find(isstart); % date major, so the segments are in order of their start date
nSeg=length(segstart);
segnodes=cell(nSeg,1);
segparent=zeros(nSeg,1);
nodeseg=zeros(nNodes,1);
for ss=1:nSeg
    kk=segstart(ss);
    chain=zeros(1,T-nodedate(kk)+1);
    chain(1)=kk;
    ll=1;
    while nodenchild(kk)==1
        kk=nodefirstchild(kk);
        ll=ll+1;
        chain(ll)=kk;
    end
    segnodes{ss}=chain(1:ll);
    segparent(ss)=nodeparent(segstart(ss));
    nodeseg(chain(1:ll))=ss;
end

Tree.N_S=N_S;
Tree.T=T;
Tree.T1=T1;
Tree.T2=T2;
Tree.S_initial=S_initial;
Tree.pi_S=pi_S;
Tree.nNodes=nNodes;
Tree.nodedate=nodedate;
Tree.nodeS=nodeS;
Tree.nodeparent=nodeparent;
Tree.nodeprob=nodeprob;
Tree.nodecondprob=nodecondprob;
Tree.nodefirstchild=nodefirstchild;
Tree.nodenchild=nodenchild;
Tree.nHist=nHist;
Tree.histories=histories;
Tree.historyprob=historyprob;
Tree.historyS=historyS;
Tree.nSeg=nSeg;
Tree.segnodes=segnodes;
Tree.segparent=segparent;
Tree.nodeseg=nodeseg;

%% Put the price and parameter paths onto the tree
PricePathNodes=struct();
if exist('PricePath','var') && ~isempty(PricePath)
    PricePathNames=fieldnames(PricePath);
    for pp=1:length(PricePathNames)
        temp=gather(PricePath.(PricePathNames{pp}));
        if isvector(temp) && numel(temp)==nNodes
            % One value per node. (If nNodes==T the tree has a single history, numbered 1,...,T, so this is also the
            % reading of a 1-by-T path.)
            PricePathNodes.(PricePathNames{pp})=reshape(temp,[1,nNodes]);
        elseif isvector(temp) && numel(temp)==T
            PricePathNodes.(PricePathNames{pp})=reshape(temp(nodedate),[1,nNodes]);
        elseif all(size(temp)==[N_S,T])
            PricePathNodes.(PricePathNames{pp})=reshape(temp(nodeS+N_S*(nodedate-1)),[1,nNodes]);
        elseif all(size(temp)==[T,N_S])
            PricePathNodes.(PricePathNames{pp})=reshape(temp(nodedate+T*(nodeS-1)),[1,nNodes]);
        else
            error('UncertainTransPathTree: PricePath.%s must be 1-by-nNodes (nNodes=%i), N_S-by-T or 1-by-T',PricePathNames{pp},nNodes)
        end
    end
end
ParamPathNodes=struct();
if exist('ParamPath','var') && ~isempty(ParamPath)
    ParamPathNames=fieldnames(ParamPath);
    for pp=1:length(ParamPathNames)
        temp=gather(ParamPath.(ParamPathNames{pp}));
        if isvector(temp) && numel(temp)==T
            ParamPathNodes.(ParamPathNames{pp})=reshape(temp(nodedate),[1,nNodes]);
        elseif all(size(temp)==[N_S,T])
            ParamPathNodes.(ParamPathNames{pp})=reshape(temp(nodeS+N_S*(nodedate-1)),[1,nNodes]);
        elseif all(size(temp)==[T,N_S])
            ParamPathNodes.(ParamPathNames{pp})=reshape(temp(nodedate+T*(nodeS-1)),[1,nNodes]);
        else
            error('UncertainTransPathTree: ParamPath.%s must be N_S-by-T (a parameter that depends on S) or 1-by-T (one that does not)',ParamPathNames{pp})
        end
    end
end

end
