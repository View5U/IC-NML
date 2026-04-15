function model = IC_NML_revised(X, Y, ~, Para)
% IC_NML (revised)
% Key changes:
%   (1) Replace post-hoc C-effect tuning with an in-training mixing operator:
%         P = (1-eta) * I + eta * Cbar
%   (2) Learn Cbar with strict constraints for interpretability/stability:
%         Cbar >= 0, diag(Cbar)=0, 1^T Cbar = 1^T  (column-stochastic, no self-loop)
%       via projected gradient (simplex projection per column).
%   (3) Add a small prior anchor towards co-occurrence prior Cbar0:
%         (mu/2) * ||Cbar - Cbar0||_F^2


Y(Y == -1) = 0;

alpha  = Para.alpha;
beta   = Para.beta;
gamma  = Para.gamma;
lambda = Para.lambda;
lr     = Para.lr;

% ---- New/extra hyperparameters (safe defaults) ----
if isfield(Para,'eta');      eta = Para.eta;      else; eta = 0.15;    end  % 0.15 correlation mixing strength in [0,1]
if isfield(Para,'mu');       mu  = Para.mu;       else; mu  = 100;     end  % 100 prior anchor weight
% if isfield(Para,'lrC');      lrC = Para.lrC;      else; lrC = 0.10;   end  % Cbar projected-gradient stepsize

rhoI = 20;

[~, num_dim] = size(X);
maxIter = Para.maxIter;
minLossMargin = Para.minLossMargin;

disp('optimizing...');

%% Initialization
W_t = (X'*X + rhoI*eye(num_dim))\(X'*Y);
W_t_1 = W_t;
V_t   = zeros(size(W_t));

% ---- Co-occurrence prior -> constrained prior Cbar0 ----
C0     = LabelCorr(Y);
Cbar0  = init_Cbar_prior(C0);
Cbar   = Cbar0;

% ---- Build mixing operator P ----
P = buildP_fast(Cbar, eta);

% ---- Dimensions (labels) ----
q  = size(Y, 2);
Iq = eye(q);

% ---- Initialize F,N (will be updated in the loop) ----
F_t = X*W_t*P;
F_t = min(max(F_t, 0), 1);
% N_t = Y - F_t;
% N_t = 0.5*(Y - F_t) + 0.1*(rand(size(F_t))-0.5);
% rho = 100;
% U_admm = zeros(size(Y));

% Instance graph: choose one from PAR/Laplacian
% L = PAR(X, 10);
L = Laplacian(X', 10);

oldloss = Inf;
iter = 1;
Loss = zeros(1, maxIter);

normXTX = norm(X'*X, 2);

%% Optimization
% ---- a/b schedule: ratio a/b linearly decays from 10 to 1 over first 10 outer iters ----
warmup_ab = 10;
a0 = 3;
a1 = 1;
b = 1;
r = 1e-1;

while iter <= maxIter
    lr = 0.975*lr;
    % Save for rollback
    W_prev = W_t;
    V_prev = V_t;
    Cbar_prev = Cbar;
    P_prev = P;

    % -------- a/b schedule --------
    if iter <= warmup_ab
        a = a0 - (iter-1) * (a0 - a1) / (warmup_ab - 1);   % iter=1 -> 10, iter=10 -> 1
    else
        a = a1;
    end

    % -------- Build P with current Cbar --------
    P = buildP_fast(Cbar, eta);

    % -------- Cache common products (VERY IMPORTANT) --------
    XW  = X * W_t;      % n x q
    XV  = X * V_t;      % n x q
    XWP = XW * P;       % n x q
    XVP = XV * P;       % n x q

    % -------- Update F (closed-form), set N = Y - F --------
    M   = (a+b)*Iq + gamma*(Iq - P)*(Iq - P)';   % q x q
    RHS = a*XWP + b*(Y - XVP);                   % n x q
    F_t = RHS / M;
    F_t = min(max(F_t, 0), 1);
    N_t = 0.5*(Y-F_t) + 0.5*(XVP) + r*(rand(size(F_t))-0.5);
    if iter > 10; r = 0;    else; r = max(1e-4, r*0.75);    end

    % ===== Update F and N (rigorous ADMM for Y = F + N) with a/b schedule =====
    %     % F-subproblem:
    %     %   min_F (a/2)||XWP - F||^2 + (gamma/2)||F - F*P||^2 + (rho/2)||Y - F - N + U||^2
    %     M   = (a + rho) * Iq + gamma * (Iq - P) * (Iq - P)';     % q x q
    %     RHS = a * XWP + rho * (Y - N_t + U_admm);                % n x q
    %     F_t = RHS / M;
    %     F_t = min(max(F_t, 0), 1);                               % optional
    %     % N-subproblem:
    %     %   min_N (b/2)||XVP - N||^2 + (rho/2)||Y - F - N + U||^2
    %     N_t = (b * XVP + rho * (Y - F_t + U_admm)) / (b + rho);  % n x q
    %     % Dual update:
    %     U_admm = U_admm + (Y - F_t - N_t);


    % -------- Update W (fast gradient, no L*X, no Proximate matrix) --------
    invrW  = inv_row_norm(W_t);  % d x 1, invrW(i)=1/||W(i,:)||2, 0 if row is all-zero
    Grad_W = GradientofW_F_fast(X, W_t, XW, XWP, F_t, P, invrW, L, a, alpha, lambda);
    stepW = Armijo_W_F_fast(X, W_t, XW, F_t, P, L, a, alpha, lambda, Grad_W, lr);
    lr = 0.5*lr + 0.5*stepW;

    W_t   = W_t_1 - stepW * Grad_W;
    W_t_1 = W_t;

    % -------- Update V (reuse XVP, avoid recompute X*V) --------
    normPPT = norm(P*(P'), 2);
    lipv = b * normXTX * normPPT;

    Grad_V = GradientofV_F_fast(X, XVP, N_t, P, b);
    H = V_t - (1/lipv) * Grad_V;
    V_t = shrinkage(H, beta/lipv);

    % -------- Update Cbar (maxIterC=1 in your setting) --------
    XW  = X * W_t;
    XV  = X * V_t;
    P = buildP_fast(Cbar, eta);

    % reuse cached multiplications as much as possible
    XWP = XW * P;
    XVP = XV * P;

    resF = XWP - F_t;          % n x q
    resN = XVP - N_t;          % n x q
    FP   = F_t * P;            % n x q
    resC = FP - F_t;           % n x q

    gradP_F    = a     * (XW') * resF;     % q x q
    gradP_N    = b     * (XV') * resN;     % q x q
    gradP_corr = gamma * (F_t') * resC;    % q x q

    gradP = gradP_F + gradP_N + gradP_corr;
    gradCbar = eta * gradP + mu * (Cbar - Cbar0);

    % -------- Lipschitz stepsize for Cbar: stepC = 1 / lipc --------
    % Tight Lipschitz constant:
    M = a*(XW'*XW) + b*(XV'*XV) + gamma*(F_t'*F_t);   % q x q, PSD
    Lp = norm(M, 2);                                  % spectral norm (largest eigenvalue)
    lipc = eta^2 * Lp + mu;
    lrC = 1 / max(lipc, 1);
    % % Cheap upper bound (always valid, often more conservative):
    %     Lp = a*norm(XW,'fro')^2 + b*norm(XV,'fro')^2 + gamma*norm(F_t,'fro')^2;
    %     lipc = eta^2 * Lp + mu;
    %     lrC = 1 / max(lipc, 1);

    Cbar = Cbar - lrC * gradCbar;
    Cbar = proj_Cbar(Cbar);

    %     % === Closed-form update for eta ===
    %     q = size(Cbar, 1);
    %     I_q = eye(q);
    %     D = Cbar - I_q;
    %     M = a*(XW'*XW) + b*(XV'*XV) + gamma*(F_t'*F_t);
    %     K = a*(XW'*F_t) + b*(XV'*N_t)  + gamma*(F_t'*F_t);
    %     MD  = M * D;
    %     den = sum(sum(D .* MD));        % <MD, D>_F = trace(D' M D), >=0
    %     num = sum(sum(D .* (K - M)));   % <K-M, D>_F
    %     eps_den = 1e-12;
    %     if den > eps_den
    %         eta_new = num / den;
    %         eta_new = min(1, max(0, eta_new));   % keep convex combination
    %     else
    %         eta_new = eta;
    %     end
    %     rho = 0.5;  % optional: smoothing to avoid oscillation
    %     eta = (1-rho)*eta + rho*eta_new;


    P = buildP_fast(Cbar, eta);

    % -------- Calculate loss (monitor) WITHOUT Proximate/trace --------
    resF = (XW * P) - F_t;
    resN = (XV * P) - N_t;
    resC = (F_t * P) - F_t;

    term_fitF  = 0.5 * a     * sum(resF(:).^2);
    term_fitN  = 0.5 * b     * sum(resN(:).^2);
    term_lcorr = 0.5 * gamma * sum(resC(:).^2);

    term_beta   = beta   * norm(V_t, 1);
    term_alpha  = alpha  * l21_norm(W_t);          % == trace(W'BW) but faster
    term_lambda = lambda * graph_trace(L, XW);     % == trace((XW)'*L*(XW)) but faster
    term_mu     = 0.5 * mu * sum((Cbar - Cbar0).^2, 'all');

    currloss = term_fitF + term_fitN + term_lcorr + term_beta + term_alpha + term_lambda + term_mu;
    Loss(iter) = currloss;

    % -------- Stop / rollback logic --------
    if (currloss > oldloss) && (iter > 10)
        % rollback and stop
        W_t   = W_prev;
        V_t   = V_prev;
        Cbar  = Cbar_prev;
        P     = P_prev;
        fprintf("No." + iter + ": loss:" + currloss + "\n");
        break;
    end
    if abs(oldloss - currloss) < minLossMargin * max(1, abs(oldloss)) && iter > 10
        fprintf("No." + iter + ": loss:" + currloss + "\n");
        break;
    end
    fprintf("No." + iter + ": loss:" + currloss + "\n");

    oldloss = currloss;
    iter = iter + 1;
end

% Final small step
% B_t = Proximate(W_t);
% Grad_W = GradientofW_F(X, W_t, F_t, P, B_t, L, a, alpha, lambda);
% stepW = Armijo_W_F(X, W_t, F_t, P, L, a, alpha, lambda, Grad_W, lr);
Grad_W = GradientofW_F_fast(X, W_t, XW, XWP, F_t, P, invrW, L, a, alpha, lambda);
stepW = Armijo_W_F_fast(X, W_t, XW, F_t, P, L, a, alpha, lambda, Grad_W, lr);
W_t = W_t_1 - Grad_W * stepW;

model.W = W_t;
model.V = V_t;
model.C = P;        % The actual mixing operator used at inference
model.Cbar = Cbar;  % Pure correlation (no self-loop), column-stochastic
model.eta = eta;
model.mu = mu;

end


%% ---------------- Helper functions ----------------

% function B = Proximate(W)
% d = size(W, 1);
% r = sqrt(sum(W.^2, 2));
% invr = zeros(d, 1);
% idx = (r > 0);
% invr(idx) = 1 ./ r(idx);
% % sparse diagonal matrix
% B = spdiags(invr, 0, d, d);
% end



% Gradient of W (use explicit F_t; no W-dependent correlation term)
% function Grad_W = GradientofW_F(X, W, F, P, B, L, a, alpha, lambda)
% Grad_W = a * (X'*(X*W*P - F) * (P')) ...
%     + 2*alpha*B*W ...
%     + lambda*(X'*L*X*W + (L*X)'*X*W);
% end



% -------- Fast Gradient of V (reuse XVP, avoid X*V inside) --------
function Grad_V = GradientofV_F_fast(X, XVP, N, P, b)
R = XVP - N;                     % n x q
Grad_V = b * (X' * R) * (P');    % d x q
end



% Soft Thresholding
function result = shrinkage(Matrix, lambda)
result = max(Matrix - lambda, 0) - max(-Matrix - lambda, 0);
end



% ---- Build constrained prior Cbar0 from co-occurrence C0 ----
function Cbar0 = init_Cbar_prior(C0)
q = size(C0, 1);
Cbar0 = C0;
Cbar0 = Cbar0 - diag(diag(Cbar0));     % diag = 0

for j = 1:q
    s = sum(Cbar0(:, j));
    if s > 0
        Cbar0(:, j) = Cbar0(:, j) / s; % column sum = 1
    else
        % if the column has no positives, use uniform mass on off-diagonals
        if q > 1
            Cbar0(:, j) = 1/(q-1);
            Cbar0(j, j) = 0;
        else
            Cbar0(:, j) = 0;
        end
    end
end

% Numerical safety: enforce constraints again
Cbar0 = proj_Cbar(Cbar0);
end



% ---- Mixing operator ----
% function P = buildP(Cbar, eta)
% q = size(Cbar, 1);
% P = (1-eta) * eye(q) + eta * Cbar;
% end



% ---- Project Cbar to: Cbar>=0, diag=0, 1^T Cbar = 1^T ----
function Cbar = proj_Cbar(Cbar)
q = size(Cbar, 1);
Cbar(1:q+1:end) = 0;

if q <= 1
    Cbar = zeros(size(Cbar));
    return;
end

for j = 1:q
    idx = true(q, 1);
    idx(j) = false;
    v = Cbar(idx, j);
    v = proj_simplex(v, 1.0);  % sum(v)=1, v>=0
    Cbar(idx, j) = v;
    Cbar(j, j) = 0;
end
end



% -------- Fast buildP: avoid eye(q) allocation each call --------
function P = buildP_fast(Cbar, eta)
q = size(Cbar, 1);
P = eta * Cbar;
P(1:q+1:end) = P(1:q+1:end) + (1 - eta);
end



% -------- inv row-norm weights for l2,1 surrogate --------
function invr = inv_row_norm(W)
r = sqrt(sum(W.^2, 2));     % d x 1
invr = zeros(size(r));
idx = (r > 0);
invr(idx) = 1 ./ r(idx);
end



% -------- l2,1 norm (used in loss / Armijo) --------
function v = l21_norm(W)
v = sum(sqrt(sum(W.^2, 2)));
end



% -------- trace(Z' * L * Z) computed via sparse multiply --------
function v = graph_trace(L, Z)
LZ = L * Z;
v = sum(sum(Z .* LZ));
end



% -------- Fast Gradient of W (no L*X, no Proximate matrix) --------
function Grad_W = GradientofW_F_fast(X, W, XW, XWP, F, P, invrW, L, a, alpha, lambda)
% Fit term
R = XWP - F;                              % n x q
Grad_fit = a * (X' * R) * (P');           % d x q

% l2,1 surrogate gradient: 2*alpha*diag(1/||row||)*W
Grad_l21 = 2 * alpha * bsxfun(@times, invrW, W);   % d x q

% Graph term: X'*(L+L')*(XW), avoid forming L*X
Z  = XW;
LZ = L  * Z;
LTZ = L' * Z;
Grad_graph = lambda * (X' * (LZ + LTZ));  % d x q

Grad_W = Grad_fit + Grad_l21 + Grad_graph;
end



% -------- Fast Armijo that avoids Proximate/trace --------
function stepsize = Armijo_W_F_fast(X, W, XW, F, P, L, a, alpha, lambda, Grad_W, lr)
c = 0.5;
stepsize = lr;
res = (XW * P) - F;
oldobj = 0.5*a*sum(res(:).^2) + alpha*l21_norm(W) + lambda*graph_trace(L, XW);

while stepsize > 1e-10
    W_new = W - stepsize * Grad_W;
    XW_new = X * W_new;

    res_new = (XW_new * P) - F;
    newobj = 0.5*a*sum(res_new(:).^2) + alpha*l21_norm(W_new) + lambda*graph_trace(L, XW_new);

    if (newobj - oldobj) <= c * sum(sum(Grad_W .* (W_new - W))) + eps
        break;
    end
    stepsize = stepsize * c;
end
end



% ---- Euclidean projection onto simplex {x>=0, sum x = z} ----
function x = proj_simplex(v, z)
% v: (m x 1), z: scalar > 0
m = length(v);
if m == 0
    x = v;
    return;
end

u = sort(v, 'descend');
cssv = cumsum(u) - z;
ind = (1:m)';
cond = u - cssv ./ ind > 0;
rho = find(cond, 1, 'last');

if isempty(rho)
    theta = cssv(end) / m;
else
    theta = cssv(rho) / rho;
end

x = max(v - theta, 0);
end



function L = LabelCorr(Y)
[~, c] = size(Y);
L = zeros(c, c);
for i = 1:c
    N = sum(Y(:, i)==1);
    for j = 1:c
        if N == 0
            L(i, j) = 0;
        else
            L(i, j) = sum(Y(:, i)'*Y(:, j))/N;
        end
    end
end
L(isnan(L)) = 0;
L(isinf(L)) = 0;
end



function L = PAR(X, k)
ins_num = size(X,1);
[~, neighbor] = pdist2(X, X, 'euclidean', 'Smallest', k+1);
neighbor = neighbor(2:end, :);
neighbor = neighbor';
rows = repmat((1:ins_num)', 1, k);
datas = zeros(ins_num, k);
for i=1:ins_num
    neighborIns = X(neighbor(i,:), :)';
    w = lsqnonneg(neighborIns, X(i,:)');
    datas(i,:) = w;
end
trans = sparse(rows, neighbor, datas, ins_num, ins_num);
sumW = full(sum(trans, 2));
sumW(sumW == 0) = 1;
A = bsxfun(@rdivide, trans, sumW);
L = diag(sum(A, 2)) - A;
end

function L = Laplacian(X, k)
options = [];
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
W = constructW(X', options);
D = diag(sum(W, 1));
L = D - W;
end
