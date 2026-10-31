function pnew=UncertainTransitionPath_InfHorz_shootingmap(p,GEc,nNodes,nPrices,nGEeqns,notlast,transpathoptions)
% The shooting map of an uncertain transition path, for Anderson acceleration in UncertainTransitionPath_InfHorz.
% p is the prices of every node stacked as nNodes-by-nPrices, as a column; GEc the general eqm conditions of the nodes
% before period T (the notlast nodes), stacked as nnotlast-by-nGEeqns, as a column.
% The map is the deterministic one, updatePricePathNew_TPath_T, applied to the nodes before period T as if they were
% the periods 1,...,T-1 of a path (it is a per-period formula, so the order of the rows does not matter), with the
% period T nodes (the final eqms) passed through unchanged. The extra row given to updatePricePathNew_TPath_T is the
% 'period T' it passes through, and is dropped.

P=reshape(p,[nNodes,nPrices]);
nNL=sum(notlast);
Pnotlast=[P(notlast,:); zeros(1,nPrices,'like',P)];
Pnotlastnew=updatePricePathNew_TPath_T(reshape(GEc,[nNL,nGEeqns]),Pnotlast,nNL+1,1,transpathoptions);
P(notlast,:)=Pnotlastnew(1:nNL,:);
pnew=reshape(P,[],1);

end
