import torch
from utils import diff_operators, quaternion

# uses real units


def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):

    # Continuous Time Loss
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output, dT, states_transitions=None):     
        if dynamics.deepReach_model == 'exact':
            dirichlet=torch.Tensor([0]) # Set Boundary Loss to 0
        else:
            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask] # Calculate Boundary Loss
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0]).cuda()
            if dynamics.deepReach_model == 'exact':
                dirichlet = output.squeeze(dim=-1)-0.0
            # print("pre training")
        else:
            # print("curriculum training")
            if dynamics.name =="Quadruped":
                ham = dynamics.hamiltonian(state, dvds, dT, states_transitions)
            else:
                ham = dynamics.hamiltonian(state, dvds, dT)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value) #PDE Loss


        return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor}


    # Discrete Time Loss
    def brt_dt_undiscounted_loss(state, value, opt_next_value, boundary_value, dirichlet_mask, output, dT):
        if dynamics.deepReach_model == 'exact':
            dirichlet = torch.Tensor([0])
        else:
            dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask] # Calculate Boundary Loss
        if torch.all(dirichlet_mask):
            # Pre-Training Loss
            diff_constraint_hom = torch.Tensor([0]).cuda()
            if dynamics.deepReach_model == 'exact':
                dirichlet = output.squeeze(dim=-1)-0.0
            # print("pre training")
        else:
            # print("curriculum training")

            diff_constraint_hom = value - torch.min(boundary_value, opt_next_value) #PDE Loss

        return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor}       

    def brt_dt_discounted_loss(state, value, opt_next_value, boundary_value, dirichlet_mask, output, dT, gamma): 
        # if dynamics.deepReach_model == 'exact':
        #     dirichlet = torch.Tensor([0])
        # else:
        #     dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask] # Calculate Boundary Loss
        dirichlet = torch.Tensor([0])
        if torch.all(dirichlet_mask):
            # Pre-Training Loss
            diff_constraint_hom = torch.Tensor([0]).cuda()
            if dynamics.deepReach_model == 'exact':
                dirichlet = output.squeeze(dim=-1)-0.0
            else:
                dirichlet = value - boundary_value # Calculate Boundary Loss
            dirichlet_loss=torch.abs(dirichlet).sum() / dirichlet_loss_divisor
            # print("pre training")
        else:
            # print("curriculum training")
            dirichlet_loss= torch.Tensor([0.0])
            diff_constraint_hom = value - (1- gamma)* boundary_value - gamma*torch.min(boundary_value, opt_next_value) #PDE Loss

        return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'dirichlet': dirichlet_loss}

    if dynamics.method_ in ["vanilla", "1AM3", "indiv", "A3", "2M1", "NN",  "model_based"]:
        return brt_hjivi_loss
    elif dynamics.method_ == "1AM1":
        return brt_dt_undiscounted_loss
    elif dynamics.method_ == "1AM2":
        return brt_dt_discounted_loss


def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output,dT):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
            discrete_bellman_loss = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds,dT)
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepReach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                diff_constraint_hom_loss = torch.abs(diff_constraint_hom).sum()
                return {'diff_constraint_hom': diff_constraint_hom_loss,
                        'discrete_bellman_loss': torch.abs(discrete_bellman_loss).sum()*0.0,
                        'dirichlet': torch.zeros_like(diff_constraint_hom_loss)}

        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'discrete_bellman_loss': torch.abs(discrete_bellman_loss).sum()*0.0}
    return brat_hjivi_loss
