import torch
import numpy as np
seed = 42
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural Generalized Predictive Control
def cost_fun_min(t_amb, u, y, ym, upp, low, model, s=1e-30):
    iteration = 3 # iter for Newton-Raphson
    for i in range(iteration):
        cfmtoms = 0.00047194745
        input = np.concatenate((y.reshape(-1, 1),
                                t_amb.reshape(-1, 1),
                                u.reshape(-1, 1),
                                np.array([424 * cfmtoms]).reshape(-1, 1)
                                ), axis=1)
        x = input.astype('float64')
        x = torch.Tensor(x).to(device)
        yp = model(x)
        yp = yp.cpu().detach().numpy()
        x.requires_grad = True
        J = torch.autograd.functional.jacobian(model, x)
        J2 = torch.autograd.functional.hessian(model, x)
        dyn_du = J[:, :, :, 2].cpu().detach().numpy()  # dy/dTevap
        d2yn_du2 = J2[:, 2, :, 2].cpu().detach().numpy()  # d^2y/dTevap^2
        lamb = 0.3  # weight number
        # J = lamb/4 * (ym(n+1)-y(n+1))**2 + (1-lamb) * 1200/1200 *((25-Tevap)/(25-10)) + g(u) # cost function
        dJ_dU = -2/4*lamb*(ym-yp)*dyn_du- s/((u - low)**2) + s/((upp - u)**2) - 1200/15*(1-lamb)/1200
        d2J_dU2 = 2/4 * lamb * ((dyn_du ** 2) - d2yn_du2 * (ym - yp)) + (2 * s) / ((u - low) ** 3) + (2 * s) / (
                        (upp - u) ** 3)
        u = u - dJ_dU / d2J_dU2  # find optimized u
        # Threshold for control input 11 <= u <= 19
        if u < 10:
            u = np.array([[[11]]])
        if u > 20:
            u = np.array([[[19]]])
    input[:, 2] = u.reshape(-1, 1)
    input = input.astype('float64')
    input = torch.Tensor(input).to(device)
    yp = model(input).cpu().detach().numpy()
    E_cost = 1200*(25-u)/(25-10)
    return u, yp, E_cost