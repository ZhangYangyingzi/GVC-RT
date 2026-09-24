"""Single fixed bottleneck. Decoder increments are anchored exactly at zero input."""
from support import torch,nn,F,Residual,Synthesis,FactorizedEntropy,CFG


class ResidualCodec(nn.Module):
    def __init__(self,residual_scale):
        super().__init__()
        a=CFG["architecture"]
        h,c=a["hidden_channels"],a["code_channels"]
        self.encoder=nn.Sequential(nn.Conv2d(36,h,3,padding=1),nn.SiLU(),Residual(h),
                                   nn.Conv2d(h,h,3,stride=2,padding=1),nn.SiLU(),Residual(h),
                                   nn.Conv2d(h,h,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(h,c,1))
        self.decoder=Synthesis(latent_channels=18,hidden_channels=h,enhancement_channels=c,
                               grid_hw=a["grid_hw"],blocks=a["blocks"])
        self.entropy=FactorizedEntropy(c)
        self.register_buffer("residual_scale",residual_scale.clone().float())

    def analyze(self,r_star,ell_c):
        u=self.encoder(torch.cat((r_star/self.residual_scale,ell_c),1))
        assert tuple(u.shape)==(1,8,17,30)
        return u

    def synthesize_normalized(self,u,ell_c):
        # Identical deterministic calls for u=0; no learned bias can create an increment.
        return self.decoder(u,ell_c)-self.decoder(torch.zeros_like(u),ell_c)

    def synthesize(self,u,ell_c):
        return self.synthesize_normalized(u,ell_c)*self.residual_scale

    def forward(self,r_star,ell_c,delta=None):
        u=self.analyze(r_star,ell_c)
        symbols=None
        if delta is not None:
            raw=u/delta
            symbols=raw+(raw.round()-raw).detach() if self.training else raw.round()
            u_hat=symbols*delta
        else:
            u_hat=u
        pred_norm=self.synthesize_normalized(u_hat,ell_c)
        return pred_norm*self.residual_scale,pred_norm,u,symbols
