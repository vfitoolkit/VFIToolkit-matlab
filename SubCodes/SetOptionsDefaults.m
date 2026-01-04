function [vfoptions, simoptions]=SetOptionsDefaults(vfoptions,simoptions)
% A one-size-fits-most approach to setting options in the VFIToolkit.
% Called without any parameters, this function will return newly created
% structures with many (but by no means all) fields typically required
% to be set in the VFOPTIONS and/or SIMOPTIONS variables used throughout
% the toolkit.
% When called with VFOPTIONS and possibly also SIMOPTIONS, set options
% not yet set without overriding options that have been set.

% heterogenous options and transpath options--you are on your own.

if exist('vfoptions','var')==0
    vfoptions=struct();
end
if ~isfield(vfoptions,'parallel')
    vfoptions.parallel=1+(gpuDeviceCount>0);
end

if exist('simoptions','var')==0
    simoptions=struct();
end

simdefaults=dictionary(...
    'parallel', vfoptions.parallel,...
    'tolerance', 10^(-12),...
    'nquantiles', 20,... % by default gives ventiles
    'npoints', 100,...
    'gridinterplayer', 0,...
    'experienceasset', 0,...
    'experienceassetu', 0,...
    nan, nan);

my_keys=keys(simdefaults);
for opt=1:numEntries(simdefaults)-1
    if ~isfield(simoptions,my_keys(opt))
        simoptions.(my_keys(opt))=lookup(simdefaults,my_keys(opt));
    end
end

