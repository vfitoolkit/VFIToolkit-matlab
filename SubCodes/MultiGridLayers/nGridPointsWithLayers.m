function RequiredGridPoints=nGridPointsWithLayers(vfoptions)

OneLayer=vfoptions.refine_pts; % points per dimension per layer

if vfoptions.refine_iter==1
    RequiredGridPoints=OneLayer;
elseif vfoptions.refine_iter==2
    TwoLayers=1+(OneLayer-1).*(OneLayer-1)/2;
    RequiredGridPoints=TwoLayers;
elseif vfoptions.refine_iter==3
    TwoLayers=1+(OneLayer-1).*(OneLayer-1)/2;
    ThreeLayers=1+(TwoLayers-1).*(OneLayer-1)/2;
    RequiredGridPoints=ThreeLayers;
elseif vfoptions.refine_iter==4
    TwoLayers=1+(OneLayer-1).*(OneLayer-1)/2;
    ThreeLayers=1+(TwoLayers-1).*(OneLayer-1)/2;
    FourLayers=1+(ThreeLayers-1).*(OneLayer-1)/2;
    RequiredGridPoints=FourLayers;
else
    error('Max of four layers (for value function refinement2)')
end

end