# Fixed method mapping

| ID | Bridge latent | Generator body | AdaGN gamma/beta maps | Role |
|---|---|---|---|---|
| A00 | original z0 | G0 | G0 | O_original endpoint |
| A01 | original z0 | G0 | BG-trained G_F | new A cross |
| A10 | original z0 | BG-trained G_F | G0 | new A cross |
| A11 | original z0 | BG-trained G_F | BG-trained G_F | Y_originalBridge_fullG endpoint |
| G_fullLoss_Gonly_500 | original z0 | independently trained G-only | independently trained G-only | B new branch |
| F_BG500 | trained zF | BG-trained G_F | BG-trained G_F | B historical BG endpoint |

A11 is Y, not F_BG500. All use the same frozen original stream and q_recon at each video/QP. The A split is a state_dict parameter intervention, not a latent-statistics input swap.
