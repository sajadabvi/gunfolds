function baselines_mvgc(workdir, n_components, alpha, momax, mhtc)
% BASELINES_MVGC  Conditional multivariate Granger causality per FBIRN subject.
%
%   baselines_mvgc(WORKDIR, N_COMPONENTS [, ALPHA, MOMAX, MHTC])
%
% Reads   <WORKDIR>/N<N_COMPONENTS>/input.mat   (written by
%         baselines_fmri_experiment.py --method MVGC --stage export):
%           data : [nsubj x T x N]   per-subject time series (time x comp)
% Writes  <WORKDIR>/N<N_COMPONENTS>/MVGC/sig_<s>.mat   (s is 0-based) each with
%           A_src_tgt : [N x N] binary,  A_src_tgt(s,t)=1  <=>  edge  s -> t
%
% Requires the MVGC toolbox (Barnett & Seth, 2014) on the MATLAB path.  The
% slurm wrapper does:  addpath(genpath(MVGC_TOOLBOX)); startup;
%
% METHOD (classic MVGC v1 time-domain pairwise-conditional GC):
%   model order p   : AIC over [1, MOMAX] (tsdata_to_infocrit), floored at 1
%   [A,SIG]         : tsdata_to_var(X, p, 'OLS')
%   G               : var_to_autocov(A, SIG)
%   F               : autocov_to_pwcgc(G)        % F(i,j) = GC from j to i
%   pval            : mvgc_pval(F, p, T, 1, 1, 1, N-2, 'F')   % conditional
%   sig             : significance(pval, alpha, mhtc)         % sig(i,j): j->i
%
% DIRECTION.  MVGC's F(i,j) is "GC to i from j", i.e. native [to, from] order.
% We TRANSPOSE before saving so Python reads source->target directly:
%             A_src_tgt = sig.'              % A_src_tgt(s,t)=1  <=>  s -> t
%
% VERSION NOTE.  The MVGC *toolbox API differs between v1.0 and MVGC2*.  This
% driver targets v1.0 (the widely-used Barnett-Seth release).  If you are on
% MVGC2, replace the F/pval block with the v2 equivalents
% (var_to_pwcgc / mvgc_cdf) -- the surrounding I/O contract (read input.mat,
% write sig_<s>.mat with A_src_tgt in [source,target]) stays identical.

    if nargin < 3 || isempty(alpha), alpha = 0.05; end
    if nargin < 4 || isempty(momax), momax = 5;    end   % bounded; T=140 is short
    if nargin < 5 || isempty(mhtc),  mhtc  = 'FDR'; end
    if ischar(n_components), n_components = str2double(n_components); end
    if ischar(alpha),        alpha        = str2double(alpha);        end
    if ischar(momax),        momax        = str2double(momax);        end

    ndir   = fullfile(workdir, sprintf('N%d', n_components));
    S      = load(fullfile(ndir, 'input.mat'));
    data   = S.data;                         % [nsubj x T x N]
    nsubj  = size(data, 1);
    N      = size(data, 3);
    outdir = fullfile(ndir, 'MVGC');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    regmode   = 'OLS';
    icregmode = 'LWR';
    tstat     = 'F';

    fprintf('MVGC: %d subjects, N=%d, momax=%d, alpha=%g, mhtc=%s\n', ...
            nsubj, N, momax, alpha, mhtc);

    for s = 1:nsubj
        X = squeeze(data(s, :, :)).';        % [N x T]  (MVGC wants vars x obs)
        X = X - mean(X, 2);                  % demean
        T = size(X, 2);
        sig = zeros(N);                      % default: no edges (on any failure)
        try
            % --- model order via AIC, bounded and floored ---
            mo = 1;
            try
                [~, ~, moAIC, ~] = tsdata_to_infocrit(X, momax, icregmode);
                if ~isempty(moAIC) && isfinite(moAIC), mo = max(1, moAIC); end
            catch
                mo = 1;
            end

            [Avar, SIG] = tsdata_to_var(X, mo, regmode);
            if any(~isfinite(Avar(:))) || any(~isfinite(SIG(:)))
                error('VAR estimation produced non-finite values');
            end

            [G, info] = var_to_autocov(Avar, SIG);
            if isfield(info, 'error') && info.error ~= 0
                error('var_to_autocov error code %d', info.error);
            end

            F = autocov_to_pwcgc(G);         % F(i,j) = GC from j to i
            % conditional pairwise GC: target dim nx=1, source dim ny=1,
            % conditioning dim nz = N-2.
            pval = mvgc_pval(F, mo, T, 1, 1, 1, N - 2, tstat);
            sigm = significance(pval, alpha, mhtc);   % NaN on diagonal
            sigm(~isfinite(sigm)) = 0;
            sig = sigm;                      % sig(i,j) = 1  <=>  j -> i
        catch ME
            fprintf('  WARN subject %d: %s -> empty graph\n', s - 1, ME.message);
        end

        A_src_tgt = double(sig.' > 0);       % [source, target]
        A_src_tgt(1:N+1:end) = 0;            % no self-loops
        save(fullfile(outdir, sprintf('sig_%04d.mat', s - 1)), 'A_src_tgt');
    end
    fprintf('MVGC: wrote %d sig files to %s\n', nsubj, outdir);
end
