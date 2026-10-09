import torch
import torch.nn.functional as F


WEIGHTS = {"rgb": 1.0, "lpips": 0.05, "dists": 0.05, "temporal": 0.20, "anchor": 0.02}


def unit(value):
    return (value.float() + 1) / 2


def pair_loss(outputs, targets, residuals, lpips_model, dists_model):
    image_terms = []
    for output, target in zip(outputs, targets):
        output_unit, target_unit = unit(output), unit(target)
        image_terms.append((F.l1_loss(output_unit, target_unit),
                            lpips_model(output_unit, target_unit, normalize=True).mean(),
                            dists_model(output_unit, target_unit, require_grad=True).mean()))
    rgb = torch.stack([term[0] for term in image_terms]).mean()
    lpips_loss = torch.stack([term[1] for term in image_terms]).mean()
    dists_loss = torch.stack([term[2] for term in image_terms]).mean()
    temporal = F.l1_loss(unit(outputs[1]) - unit(outputs[0]), unit(targets[1]) - unit(targets[0]))
    anchor = torch.stack([residual.abs().mean() for residual in residuals]).mean()
    total = (WEIGHTS["rgb"] * rgb + WEIGHTS["lpips"] * lpips_loss +
             WEIGHTS["dists"] * dists_loss + WEIGHTS["temporal"] * temporal +
             WEIGHTS["anchor"] * anchor)
    return total, {"total": float(total.detach()), "rgb": float(rgb.detach()),
                   "lpips": float(lpips_loss.detach()), "dists": float(dists_loss.detach()),
                   "temporal": float(temporal.detach()), "anchor": float(anchor.detach())}
