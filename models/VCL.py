# @author = "avirambh"
# @email = "aviramb@mail.tau.ac.il"

import torch
import torch.nn as nn


def print_vcls(vcls, epoch):
    for vcl in vcls:
        print("Step {}: Name: {} Loss: {}".format(epoch,
                                                  vcl.name,
                                                  vcl.get_layer_loss()))

def get_vcl_loss(model, epoch, debug):
    if debug and epoch % 5 == 0:
        print_vcls(model.vcls, epoch)

    vcl_loss = VCL.get_forward_loss()
    VCL.zero_loss()
    return vcl_loss


def apply_vcl(model, tmp_input, sample_size=5, eps_learn=True):

    # Init VCL hooks
    vcls = init_vcl(model)

    init_tensor = torch.tensor(tmp_input, dtype=torch.float32).unsqueeze(0)
    model.forward(init_tensor)

    # Register beta parameters
    for ix, vcl in enumerate(vcls):
        name = 'vcl_{}'.format(ix)
        model.register_parameter(name, vcl.vcl_beta)
        vcl.set_name(name)
        vcl.set_sample_size(sample_size)
        vcl.vcl_beta.requires_grad = eps_learn

    return vcls


def init_vcl(model, pre_activation=True):
    vcls = []
    for child in model.children():
        cur_vcls = init_vcl(child)
        if type(cur_vcls) == VCL:
            vcls.append(cur_vcls)
        elif cur_vcls:
            vcls.extend(cur_vcls)

    isActivation = type(model).__module__ == 'torch.nn.modules.activation'
    isVCL = not (hasattr(model, 'no_vcl') and model.no_vcl)

    if isActivation:
        print("Adding VCL forward hook to {}: {}".format(type(model), isVCL))
        if isVCL:
            cur_vcl = VCL()
            if pre_activation:
                model.register_forward_pre_hook(cur_vcl)
            else:
                model.register_forward_hook(cur_vcl)

            model.inplace = False
            return cur_vcl
    return vcls


class VCL(nn.Module):
    '''
    Implementing VCL. This loss is called usually before or after an activation
    and calculated per a given input.
    '''

    forward_loss = []
    step = 0

    def __init__(self, device='cuda:0', beta_init=1.0, sample_size=5, name='vcl'):
        super(VCL, self).__init__()
        self.layer_loss = 0
        self.initialized = False
        self.beta_init = beta_init
        self.sample_size = sample_size
        self.device = device
        self.name = name

    @classmethod
    def get_forward_loss(self):
        VCL.step += 1
        return sum(VCL.forward_loss)

    @classmethod
    def zero_loss(self):
        VCL.forward_loss = []

    def set_name(self, name):
        self.name = name

    def set_sample_size(self, sample_size):
        self.sample_size = sample_size

    def get_layer_loss(self):
        return self.layer_loss

    def forward(self, *inp):

        # Unpack when VCL is called in a forward hook
        if len(inp) > 1:
            inp = inp[1][0]
        # Unpack when VCL is a normal layer
        else:
            inp = inp[0] #TODO: change name

        # Initialization pass
        if not self.initialized:
            N,C,H,W = inp.shape
            tmp_tensor = torch.ones([C, H, W],
                                    device=self.device, dtype=torch.float32) * self.beta_init
            self.vcl_beta = nn.Parameter(tmp_tensor) # Requires grad True by default, but need to init before optimizer
            self.initialized = True
            return input

        # Slice
        slices = torch.split(inp, self.sample_size, dim=0)

        # Calc variance per activation
        offset = 0
        var_a = slices[offset].var(dim=0).abs()
        var_b = slices[offset+1].var(dim=0).abs()

        # Calculate VCL term
        with torch.enable_grad():
            self.layer_loss = (1-var_a/(var_b + self.vcl_beta)).pow(2).mean()
            VCL.forward_loss.append(self.layer_loss)

        # Verify stable loss
        if torch.isnan(self.layer_loss):
            print("VCL exploded!")
            exit(1)

        return inp