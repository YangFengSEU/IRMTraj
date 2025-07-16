from torch import nn, optim, autograd


def get_optimizer(model ,normal_op, score_op):
    if normal_op =="adam" and score_op =="adam":
        parameters = list(model.named_parameters())
        for n, v in parameters:
            if ("score" not in n) and v.requires_grad:
                print(n, "weight_para")
        for n, v in parameters:
            if ("score" in n) and v.requires_grad:
                print(n, "score_para")
        weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
        score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
        optimizer1 = optim.Adam(
            score_params, lr=6e-3, weight_decay= 0
        )
        optimizer2 = optim.Adam(
            weight_params,
            8e-4,
        )
        # print("opt1, opt2", optimizer1, optimizer2)
        return optimizer1, optimizer2
    return None, None

def solve_v_total(model, total):
    k = total * 0.95
    a, b = 0, 0
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            b = max(b, m.scores.max())
    def f(v):
        s = 0
        for n, m in model.named_modules():
            if hasattr(m, "scores") and m.prune:
                s += (m.scores - v).clamp(0, 1).sum()
        return s - k
    if f(0) < 0:
        return 0, 0
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v, itr

def constrainScoreByWhole(model):
    total = 0
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                total += m.scores.nelement()

    v, itr = solve_v_total(model, total)

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                m.scores.sub_(v).clamp_(0, 1)
