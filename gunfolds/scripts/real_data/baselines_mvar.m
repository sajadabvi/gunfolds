function baselines_mvar(workdir, n_components, alpha, p, mhtc)
% BASELINES_MVAR  Multivariate-AR connectivity per FBIRN subject (MATLAB bridge).
%
%   baselines_mvar(WORKDIR, N_COMPONENTS [, ALPHA, P, MHTC])
%
% Reads   <WORKDIR>/N<N_COMPONENTS>/input.mat   (written by
%         baselines_fmri_experiment.py --method MVAR --stage export):
%           data : [nsubj x T x N]   per-subject time series (time x comp)
% Writes  <WORKDIR>/N<N_COMPONENTS>/MVAR/sig_<s>.mat   (s is 0-based) each with
%           A_src_tgt : [N x N] binary,  A_src_tgt(s,t)=1  <=>  edge  s -> t
%
% METHOD.  Fit a VAR(p) to each subject by OLS (one equation per target i):
%             x_i(t) = c_i + sum_{k=1..p} sum_{j} A(i,j,k) x_j(t-k) + e_i(t)
% A directed edge  j -> i  is declared when the *block* of p lag coefficients
% {A(i,j,1),...,A(i,j,p)} is jointly significant by a Wald chi^2 test
% (H0: all p coefficients are zero).  This is the standard "significant MVAR
% influence" the legacy MATLAB pipeline produced as the `sig` matrix.
%
% DIRECTION.  The VAR coefficient A(i,j,:) is the effect of source j on target
% i, so the native significance matrix `sig(i,j)` is in [to, from] order.  We
% TRANSPOSE before saving so Python reads source->target directly:
%             A_src_tgt = sig.'              % A_src_tgt(s,t)=1  <=>  s -> t
%
% Self-contained: needs only base MATLAB + Statistics Toolbox (chi2cdf).  If
% chi2cdf is unavailable, a gammainc-based fallback is used.
%
% This mirrors the I/O contract of baselines_mvgc.m; the two are
% interchangeable from Python's point of view (both emit sig_<s>.mat).

    if nargin < 3 || isempty(alpha), alpha = 0.05; end
    if nargin < 4 || isempty(p),     p     = 1;    end   % FBIRN: short T(=140)
    if nargin < 5 || isempty(mhtc),  mhtc  = 'FDR'; end  % 'none' | 'FDR' | 'Bonferroni'
    if ischar(n_components), n_components = str2double(n_components); end
    if ischar(alpha),        alpha        = str2double(alpha);        end
    if ischar(p),            p            = str2double(p);            end

    ndir   = fullfile(workdir, sprintf('N%d', n_components));
    S      = load(fullfile(ndir, 'input.mat'));
    data   = S.data;                         % [nsubj x T x N]
    nsubj  = size(data, 1);
    N      = size(data, 3);
    outdir = fullfile(ndir, 'MVAR');
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    fprintf('MVAR: %d subjects, N=%d, p=%d, alpha=%g, mhtc=%s\n', ...
            nsubj, N, p, alpha, mhtc);

    for s = 1:nsubj
        X = squeeze(data(s, :, :));          % [T x N]
        sig = mvar_sig_one(X, p, alpha, mhtc);   % sig(i,j)=1 <=> j -> i
        A_src_tgt = double(sig.' > 0);       % [source, target]
        A_src_tgt(1:N+1:end) = 0;            % no self-loops
        save(fullfile(outdir, sprintf('sig_%04d.mat', s - 1)), 'A_src_tgt');
    end
    fprintf('MVAR: wrote %d sig files to %s\n', nsubj, outdir);
end


function sig = mvar_sig_one(X, p, alpha, mhtc)
% Fit VAR(p) by OLS, Wald-test each source-block of lag coefficients.
% Returns sig(i,j) = 1 when source j -> target i is significant ([to,from]).
    [T, N] = size(X);
    X = X - mean(X, 1);                      % demean each variable
    Teff = T - p;
    if Teff <= N * p + 1
        sig = zeros(N);                      % not enough samples to fit
        return;
    end

    % Design matrix Z: [Teff x (N*p + 1)], columns = [lag1 vars ... lagp vars, 1]
    Z = ones(Teff, N * p + 1);
    for k = 1:p
        Z(:, (k-1)*N + (1:N)) = X(p - k + 1 : T - k, :);
    end
    Y = X(p + 1 : T, :);                     % [Teff x N] targets

    ZtZ = Z.' * Z;
    ZtZ_inv = pinv(ZtZ);                     % robust to near-collinearity
    B = ZtZ_inv * (Z.' * Y);                 % [(N*p+1) x N] coeffs, col i = eq i
    resid = Y - Z * B;                       % [Teff x N]
    dof = Teff - (N * p + 1);

    pvals = ones(N);                         % pvals(i,j); diagonal stays 1
    for i = 1:N                              % target equation i
        s2 = (resid(:, i).' * resid(:, i)) / dof;   % residual variance
        Vbeta = s2 * ZtZ_inv;               % coeff covariance for equation i
        bi = B(:, i);
        for j = 1:N                          % candidate source j
            if i == j, continue; end
            idx = ((0:p-1) * N) + j;         % the p lag coeffs of j in eq i
            bj  = bi(idx);
            Vjj = Vbeta(idx, idx);
            W   = bj.' * pinv(Vjj) * bj;     % Wald stat ~ chi^2(p)
            pvals(i, j) = 1 - chi2cdf_safe(W, p);
        end
    end

    sig = apply_mhtc(pvals, alpha, mhtc, N);
end


function sig = apply_mhtc(pvals, alpha, mhtc, N)
% Threshold an [N x N] p-value matrix (diagonal ignored) with optional
% multiple-hypothesis correction over the N*(N-1) off-diagonal tests.
    mask = ~eye(N) > 0;
    pv = pvals(mask);
    m = numel(pv);
    switch lower(mhtc)
        case 'none'
            keep = pv < alpha;
        case 'bonferroni'
            keep = pv < (alpha / m);
        case 'fdr'                           % Benjamini-Hochberg
            [ps, ord] = sort(pv);
            thr = alpha * (1:m).' / m;
            below = find(ps <= thr, 1, 'last');
            keep = false(m, 1);
            if ~isempty(below)
                keep(ord(1:below)) = true;
            end
        otherwise
            keep = pv < alpha;
    end
    sig = zeros(N);
    sig(mask) = keep;
end


function c = chi2cdf_safe(x, k)
% chi2cdf if available, else lower regularised incomplete gamma.
    if exist('chi2cdf', 'file') == 2 || exist('chi2cdf', 'builtin') == 5
        c = chi2cdf(x, k);
    else
        c = gammainc(x / 2, k / 2);          % == chi2cdf(x,k)
    end
end
