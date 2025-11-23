"""ReImplement Correlation Module

James Chan
@article{MaskCRT,
  title={MaskCRT: Masked Conditional Residual Transformer for Learned Video Compression},
  author={Chen, Yi-Hsin and Xie, Hong-Sheng and Chen, Cheng-Wei and Gao, Zong-Lin and Benjak, Martin and Peng, Wen-Hsiao and Ostermann, J{\"o}rn},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3427426}
  }
"""
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn
from .functional import center_of, getWH, inv3x3, meshgrid



def linearized_grid_sample(input, grid, padding_mode='zeros', align_corners=False,
                           num_grid=8, noise_strength=.5, need_push_away=True, fixed_bias=False):
    """Linearized multi-sampling

    Args:
        input (tensor): (B, C, H, W)
        grid (tensor): (B, H, W, 2)
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        num_grid (int, optional): multisampling. Defaults to 8.
        noise_strength (float, optional): auxiliary noise. Defaults to 0.5.
        need_push_away (bool, optional): pushaway grid. Defaults to True.
        fixed_bias (bool, optional): Defaults to False.

    Returns:
        tensor: linearized sampled input

    Reference:
        paper: https://arxiv.org/abs/1901.07124
        github: https://github.com/vcg-uvic/linearized_multisampling_release
    """
    LinearizedMutilSample.set_hyperparameters(
        num_grid=num_grid, noise_strength=noise_strength, need_push_away=need_push_away, fixed_bias=fixed_bias)
    return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)


@torch.jit.script
def u(s, a: float = -0.75):
    s2, s3 = s**2, s**3
    l1 = (a+2)*s3 - (a+3)*s2 + 1
    l2 = a*s3 - (5*a)*s2 + (8*a)*s - 4*a
    return l1.where(s <= 1, l2)


@torch.jit.script
def bicubic_grid_sample(input, grid, padding_mode: str = 'zeros', align_corners: bool = False):
    """bicubic_grid_sample"""
    kernel_size = 4
    if not align_corners:
        grid = grid * getWH(input) / getWH(input).sub_(1)
    center = center_of(input)
    abs_loc = ((grid + 1) * center).unsqueeze(-1)

    locs = abs_loc.floor() + torch.tensor([-1, 0, 1, 2], device=grid.device)

    loc_w, loc_h = locs.detach().flatten(0, 2).unbind(dim=-2)
    loc_w = loc_w.reshape(-1, 1, kernel_size).expand(-1, kernel_size, -1)
    loc_h = loc_h.reshape(-1, kernel_size, 1).expand(-1, -1, kernel_size)
    loc_grid = torch.stack([loc_w, loc_h], dim=-1)
    loc_grid = loc_grid.view(grid.size(0), -1, 1, 2)/center - 1

    selected = F.grid_sample(input, loc_grid.detach(), mode='nearest',
                             padding_mode=padding_mode, align_corners=True)
    patch = selected.view(input.size()[:2]+grid.size()[1:3]+(kernel_size,)*2)

    mat_r, mat_l = u(torch.abs(abs_loc - locs.detach())).unbind(dim=-2)
    output = torch.einsum('bhwl,bchwlr,bhwr->bchw', mat_l, patch, mat_r)
    return output


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    original function prototype:
    torch.nn.functional.grid_sample(
        input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    copy from pytorch 1.3.0 source code
    add linearized_grid_sample and bicubic_grid_sample
    """
    if mode == 'linearized':
        assert input.dim() == grid.dim() == 4
        return LinearizedMutilSample.apply(input, grid, padding_mode, align_corners)
    if mode == 'bicubic':
        assert input.dim() == grid.dim() == 4
        return bicubic_grid_sample(input, grid, padding_mode, align_corners)
    else:
        return F.grid_sample(input, grid, mode, padding_mode, align_corners)


def homography_grid(matrix, size, align_corners=True):
    # type: (Tensor, List[int]) -> Tensor
    grid = cat_grid_z(meshgrid(size, align_corners,
                               device=matrix.device))  # B, H, W, 3
    homography = grid.flatten(1, 2).bmm(matrix.transpose(1, 2)).view_as(grid)
    grid, ZwarpHom = homography.split([2, 1], dim=-1)
    return grid / ZwarpHom.add(1e-8)

def transform_grid(matrix, size, align_corners=True):
    if matrix.size()[1:] in [(2, 3), (3, 4)]:
        return F.affine_grid(matrix, size, align_corners)
    else:
        return homography_grid(matrix, size, align_corners)

def affine(input, theta, size=None, sample_mode='bilinear', padding_mode='border', align_corners=False):
    # type: (Tensor, Tensor, Optional[List[int]], str, str, bool) -> Tensor
    """SPT affine function

    Args:
        input: 4-D tensor (B, C, H, W) or 5-D tensor (B, C, D, H, W)
        theta: 3-D tensor (B, 2, 3) or (B, 3, 4)
        size (Size): output size. Default: input.size()
    """
    assert input.dim() in [4, 5] and theta.dim() == 3
    assert input.size(0) == theta.size(
        0), 'batch size of inputs do not match the batch size of theta'
    if size is None:
        size = input.size()[2:]
    size = (input.size(0), 1) + tuple(size)
    return grid_sample(input, transform_grid(theta, size, align_corners), sample_mode, padding_mode, align_corners)


def shift(input, motion, size=None, sample_mode='bilinear', padding_mode='border', align_corners=False):
    """SPT shift function

    Args:
        input: 4-D tensor (B, C, H, W) or 5-D tensor (B, C, D, H, W)
        motion (motion): motion (B, 2) or (B, 3)
    """
    B = motion.size(0)
    MD = input.dim() - 2

    defo = torch.eye(MD).to(input.device)
    txy = motion.view(B, MD) / center_of(input)
    theta = torch.cat([defo.expand(B, MD, MD), txy.view(B, MD, 1)], dim=2)
    return affine(input, theta, size, sample_mode, padding_mode, align_corners)
class Correlation(Module):
    """Correlation metion in `FlowNet`.

    Args:
        num_input (int): input numers. Default: 2
        kernel_size (int or pair of int): Default: 21
        dilation (int or pair of int): correlation to larger displacement. Default: 1
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'

    Shape:
        - Input: :math:`(B, N, C, H, W)`
        - Output: :math:`(B, (N-1)K, H, W)` where `K` means kernel area
    """

    def __init__(self, num_input=2, kernel_size=21, dilation=1, padding_mode='zeros'):
        super(Correlation, self).__init__()
        self.num_input = num_input
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation
        self.padding_mode = padding_mode

        grid = meshgrid((1, 1, kernel_size[0], kernel_size[1]))
        scale = grid.new_tensor([kernel_size[0]//2*dilation[0],
                                 kernel_size[1]//2*dilation[1]])
        self.grid = grid.flatten(1, 2)*scale
        self.sample_kwargs = dict(
            sample_mode='nearest', padding_mode=self.padding_mode, align_corners=True)

    def extra_repr(self):
        return '{num_input}, kernel_size={kernel_size}, dilation={dilation}, padding_mode={padding_mode}'.format(**self.__dict__)

    def forward(self, *inputs):  # -> (B, K(N-1), H, W)
        if inputs[0].dim() == 4:
            inputs = torch.stack(inputs, dim=1)
        else:
            inputs = inputs[0]
        B, N = inputs.size()[:2]
        K = self.grid.size(1)
        assert N == self.num_input
        inputs = inputs.detach().mul(1e10).floor().div(1e10) - inputs.detach() + inputs

        target = inputs[:, -1]  # take last input as correlation target
        grid = self.grid.to(inputs.device).expand(B, -1, -1)

        return torch.stack([(inputs[:, :N-1]*shift(target, grid[:, k], **self.sample_kwargs).unsqueeze(1)).mean(-3)
                            for k in range(K)], dim=1).flatten(1, 2)