function model = IC_NML(X, Y, Para)
Y(Y == -1) = 0;
gamma = Para.gamma;
beta  = Para.beta;
alpha = Para.alpha;
lambda = Para.lambda;
lr = Para.lr;
rhoI = 16;
[~, num_dim]=size(X);
maxIter = Para.maxIter;
minLossMargin = Para.minLossMargin;
disp('optimizing...');

%% Initialization
W_t = (X'*X + rhoI*eye(num_dim))\(X'*Y);
W_t_1 = W_t;
V_t = zeros(size(W_t));
C_t = LabelCorr(Y);
C_t = resizeC(C_t);
InsCorr = PAR(X, 20);
L = diag(sum(InsCorr, 2)) - InsCorr;
oldloss = Inf;
iter = 1;
Loss = zeros(1, maxIter);
normXTX = norm(X'*X, 2);
normCCT = norm(C_t*(C_t'), 2);
lipv = normXTX*normCCT;

%% Optimization
while iter <= maxIter
    B_t = Proximate(W_t);
    Grad_W = GradientofW(X,W_t,Y,V_t,C_t,B_t,L,gamma,alpha,lambda);
    W_t = W_t_1 - Grad_W*Armijo_W(X,W_t,Y,V_t,C_t,L,gamma,alpha,lambda,Grad_W, lr);
    W_t_1 = W_t;

    % ISTA
    Grad_V = GradientofV(X, W_t, Y, V_t, C_t);
    H = V_t - (1/lipv)*Grad_V;
    V_t = shrinkage(H, beta/lipv);

    XW = X*W_t;
    XV = X*V_t;
    F = XW*C_t;

    C_t_1 = C_t;
    C_t = ((XW+XV)'*(XW+XV) + gamma*(F)'*F + rhoI*eye(size(C_t))) \ (gamma*(F)'*F + (XW+XV)'*Y);
    C_t = C_t - diag(diag(C_t)) + eye(size(C_t));

    % Calculate loss
    term_lse = 0.5*(norm(X*(W_t+V_t)*C_t-Y, 'fro')^2);
    term_lcorr = 0.5*gamma*norm(F-F*C_t, 'fro')^2;
    term_beta  = beta*norm(V_t, 1);
    term_alpha = alpha*trace(W_t'*B_t*W_t);
    term_lambda  = lambda*trace((X*W_t)'*L*(X*W_t));
    currloss = term_lse + term_lcorr + term_beta + term_alpha + term_lambda;
    Loss(iter) = currloss;

    % stop function
    if (oldloss < currloss)
        C_t = pinv((XW+XV)'*(XW+XV)) * ((XW+XV)'*Y);
        term_lse = 0.5*(norm(X*(W_t+V_t)*C_t-Y, 'fro')^2);
        term_lcorr = 0.5*gamma*norm(F-F*C_t, 'fro')^2;
        term_beta  = beta*norm(V_t, 1);
        term_alpha = alpha*trace(W_t'*B_t*W_t);
        term_lambda  = lambda*trace((X*W_t)'*L*(X*W_t));
        currloss = term_lse + term_lcorr + term_beta + term_alpha + term_lambda;
        Loss(iter) = currloss;
    end
    if (oldloss < currloss && iter > 10)
        C_t = C_t_1;
        W_t = W_t_1;
        break;
    end
    if abs(oldloss - currloss) < minLossMargin*abs(oldloss) && iter > 10
        break;
    end
    oldloss = currloss;
    iter = iter + 1;
end

B_t = Proximate(W_t);
Grad_W = GradientofW(X,W_t,Y,V_t,C_t,B_t,L,gamma,alpha,lambda);
W_t = W_t_1 - Grad_W*lr;
model.W = W_t;
model.V = V_t;
model.C = C_t;

end





%% Functions

function B = Proximate(W)
num = size(W, 1);
B = zeros(num, num);
for i = 1:num
    temp = norm(W(i, :), 2);
    if temp ~= 0
        B(i, i) = 1/temp;
    else
        B(i, i) = 0;
    end
end
end

% Gradient of W
function Grad_W = GradientofW(X,W,Y,V,C,B,L,gamma,alpha,lambda)
Grad_W = X'*(X*(W+V)*C-Y)*(C') + gamma*( X'*(X*W*C*C-X*W*C)*(C')*(C') - X'*(X*W*C*C-X*W*C)*C' ) + 2*alpha*B*W + lambda*(X'*L*X*W + (L*X)'*X*W);
end

% Gradient of V
function Grad_V = GradientofV(X,W,Y,V,C)
Grad_V = X'*(X*(W+V)*C-Y)*C';
end

% Soft Thresholding
function result = shrinkage(Matrix, lambda)
result = max(Matrix - lambda,0) - max(-Matrix - lambda,0);
end

% Armijo
function stepsize = Armijo_W(X, W, Y, V, C, L, gamma, alpha, lambda, Grad_W, lr)
c = 0.25;
stepsize = lr;
B = Proximate(W);
W_new = W - stepsize*Grad_W;
B_new = Proximate(W_new);
oldobj = 0.5*(norm(X*(W+V)*C-Y, 'fro')^2) + 0.5*gamma*norm(X*W*C*C-X*W*C, 'fro')^2 + alpha*trace(W'*B*W) + lambda*trace((X*W)'*L*(X*W));
newobj = 0.5*(norm(X*(W_new+V*C)-Y, 'fro')^2) + 0.5*gamma*norm(X*W_new*C*C-X*W_new*C, 'fro')^2 + alpha*trace(W_new'*B_new*W_new) + lambda*trace((X*W_new)'*L*(X*W_new));
if (newobj - oldobj) > c*sum(sum(Grad_W.*(W_new-W)))
    while stepsize>1e-10
        stepsize = stepsize*c;
        W_new = W - stepsize*Grad_W;
        newobj = 0.5*(norm(X*(W_new+V*C)-Y, 'fro')^2) + 0.5*gamma*norm(X*W_new*C*C-X*W_new*C, 'fro')^2 + alpha*trace(W_new'*B_new*W_new) + lambda*trace((X*W_new)'*L*(X*W_new));
        if (newobj - oldobj) <= c*sum(sum(Grad_W.*(W_new-W))) + eps
            break;
        end
    end
else
    return;
end
end

function L = LabelCorr(Y)
[~, c] = size(Y);
L = zeros(c, c);
for i = 1:c
    N = sum(Y(:, i)==1);
    for j = 1:c
        L(i, j) = sum(Y(:, i)'*Y(:, j))/N;
    end
end
L(isnan(L)) = 0;
L(isinf(L)) = 0;
end

function C = resizeC(C)
tmpC = C - diag(diag(C));
if( any(tmpC(:)>0.5) )
    diagC = diag(diag(C));
    C = C - diag(diag(C));
    for i=1:size(C,1)
        scale = sum(C(:,i));
        if scale ~= 0
            C(:,i) = C(:,i)/scale;
        end
    end
    C = C + diagC;
end
end

function A = PAR(X, k)
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
trans = bsxfun(@rdivide, trans, sumW);
A = full(trans);
end