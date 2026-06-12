function [Xcorr, alphaHat, info] = paretoTopcodeCorrect(X, opts)
% Pareto-tail correction for survey income/wealth series with right-tail
% censoring. For each column of X (treated as one cross-section / year),
% the right-tail Pareto index is estimated from the non-top-coded upper
% tail and the censored observations are replaced with their conditional
% Pareto mean.
%
% This is the procedure described in Kaplan (2012) Appendix A: "I deal
% with top-coded observations by assuming that the underlying
% distribution for each component of income is Pareto, and by
% forecasting the mean value of the top-coded observations by
% extrapolating a Pareto density fitted to the non-top-coded end of the
% observed distribution."
%
% Inputs
%   X       N x T matrix. Rows are units (households), columns are
%           cross-sections (PSID waves). NaN entries are ignored.
%   opts    optional struct:
%             .topcodes     1 x T vector of pre-specified top-code values
%                           (use NaN to auto-detect that column).
%                           Default: all NaN (auto-detect everywhere).
%             .lowerPct     percentile used as the lower threshold for the
%                           Pareto fit, computed over non-top-coded
%                           positive observations (default: 70).
%             .minObsAtCap  minimum mass at the column maximum needed to
%                           treat it as a topcode in auto-detect mode
%                           (default: 3).
%             .minFitObs    minimum number of non-top-coded observations
%                           above the lowerPct threshold needed to fit
%                           Pareto (default: 20). If fewer, falls back to
%                           opts.alphaFloor.
%             .alphaFloor   fallback Pareto index when the fit fails or
%                           returns alpha <= 1 (default: 1.5).
%             .verbose      0/1, print one diagnostic line per corrected
%                           column (default: 0).
%
% Outputs
%   Xcorr     N x T matrix with topcoded entries replaced by
%             topcode * alpha/(alpha-1).
%   alphaHat  1 x T vector of estimated tail indices per column; NaN
%             where no top-coding was detected or correction was skipped.
%   info      diagnostic struct with fields:
%               .topcodes  1 x T  top-code value used per column
%               .nAtCap    1 x T  number of observations replaced
%               .nFit      1 x T  fit-sample size per column
%               .corrMean  1 x T  imputed conditional mean per column
%
% Method (per column)
%   1. Pick the topcode: either opts.topcodes(t) if supplied, or the
%      column maximum provided at least opts.minObsAtCap observations sit
%      there (a heuristic for a hard censoring boundary).
%   2. Select the non-top-coded upper tail: positive observations strictly
%      below the topcode, above the lowerPct percentile of that subset.
%   3. Fit Pareto by Hill MLE: alpha = 1 / mean( log( x_i / x_low ) ).
%      If alpha <= 1 or the sample is too small, use opts.alphaFloor.
%   4. Replace each top-coded entry with topcode * alpha / (alpha - 1).
%
% Loops only over columns; each per-column step is vectorised.

if nargin<2 || isempty(opts), opts = struct(); end
if ~isfield(opts,'lowerPct'),    opts.lowerPct    = 70;   end
if ~isfield(opts,'minObsAtCap'), opts.minObsAtCap = 3;    end
if ~isfield(opts,'minFitObs'),   opts.minFitObs   = 20;   end
if ~isfield(opts,'alphaFloor'),  opts.alphaFloor  = 1.5;  end
if ~isfield(opts,'verbose'),     opts.verbose     = 0;    end

[~, T] = size(X);
if ~isfield(opts,'topcodes') || isempty(opts.topcodes)
    opts.topcodes = nan(1, T);
end

Xcorr    = X;
alphaHat = nan(1, T);
info     = struct( ...
    'topcodes', nan(1,T), ...
    'nAtCap',   zeros(1,T), ...
    'nFit',     zeros(1,T), ...
    'corrMean', nan(1,T));

for tt = 1:T
    x  = X(:, tt);
    ok = ~isnan(x) & x > 0;        % positive, non-missing
    if nnz(ok) < 50
        continue
    end

    %% Step 1: resolve the topcode for this column
    tc = opts.topcodes(tt);
    if isnan(tc)
        xv = x(ok);
        tc_candidate = max(xv);
        nAtMax = sum(xv == tc_candidate);
        if nAtMax < opts.minObsAtCap
            continue                % no apparent top-coding
        end
        tc = tc_candidate;
    end

    capMask   = ok & (x == tc);     % rows to be replaced
    belowMask = ok & (x <  tc);     % candidate fit data

    %% Step 2: select the upper-tail fit sample
    if nnz(belowMask) < opts.minFitObs
        continue
    end
    xb     = x(belowMask);
    x_low  = prctile(xb, opts.lowerPct);
    fitMask = belowMask & (x >= x_low);
    xf      = x(fitMask);
    nFit    = numel(xf);

    %% Step 3: Hill MLE for Pareto tail index
    if nFit < opts.minFitObs
        alpha = opts.alphaFloor;
    else
        alpha = 1 / mean(log(xf / x_low));
        if ~isfinite(alpha) || alpha <= 1
            alpha = opts.alphaFloor;
        end
    end

    %% Step 4: replace censored values with the conditional Pareto mean
    corrMean = tc * alpha / (alpha - 1);
    Xcorr(capMask, tt) = corrMean;

    %% Bookkeeping
    alphaHat(tt)      = alpha;
    info.topcodes(tt) = tc;
    info.nAtCap(tt)   = nnz(capMask);
    info.nFit(tt)     = nFit;
    info.corrMean(tt) = corrMean;

    if opts.verbose
        fprintf(['paretoTopcodeCorrect: col %2d  topcode=%10.0f  ', ...
                 'nAtCap=%4d  nFit=%5d  alpha=%5.2f  corrMean=%10.0f\n'], ...
                 tt, tc, info.nAtCap(tt), nFit, alpha, corrMean);
    end
end

end
