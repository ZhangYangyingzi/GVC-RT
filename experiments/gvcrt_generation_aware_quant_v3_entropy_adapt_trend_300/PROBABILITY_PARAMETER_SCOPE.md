# Probability parameter scope

N1 and Q1 open exactly the same five original modules: `pm.hyper_dec`, `pm.temporal_prior_encoder`, `pm.y_prior_fusion`, `pm.y_spatial_prior`, `pm.bit_estimator_z`. No V is installed.

`y_prior_fusion` jointly predicts the original local quant_step, scales and means. All original outputs may adapt in N1 and Q1. N1 means original quantization structure without the new H; it does **not** mean a fixed local quantization step.

Trainable G: the original `pm.recon_generation_net.decoder`. New H: `pm.quant_adjust` only in Q1, with exactly the v2 H11 step0 state (m=1). I codec, enc/hyper_enc, dec, feature extractor/adaptors, Bridge (`recon_generation_net.mlp`), q_scale parameters and evaluation networks stay frozen. Freezing these parameters does not disable autograd through their supervised-frame forwards.

All master parameters and accumulated gradients are FP32. The five Phi modules are initialized from the original codec FP32 tensors. Y-network forwards use differentiable FP16 parameter views; BitEstimator Bitparm forwards use FP16 parameter views with inherited FP32 CDF inputs. The same views are used by training, compress and decompress; no optimizer updates a half-precision parameter. The Generator remains at the v2 FP32 forward and v2 initial rounding.

Complete Phi state_dicts are saved. Every model load clears decode caches, creates a fresh entropy coder and rebuilds integer CDFs **after** restoring Phi. Forward probability-parameter and integer-CDF hashes must agree between encoder and fresh decoder. Gaussian table/RANS algorithms are unchanged. Training uses live differentiable CDF probabilities, not those integer tables.

|Exact parameter name (same in N1/Q1)|Shape|Master dtype|Count|
|---|---|---|---:|
|pm.bit_estimator_z.f1.h|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f1.b|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f1.a|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f2.h|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f2.b|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f2.a|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f3.h|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f3.b|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f3.a|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f4.h|[12, 128, 1, 1]|torch.float32|1536|
|pm.bit_estimator_z.f4.b|[12, 128, 1, 1]|torch.float32|1536|
|pm.hyper_dec.conv.0.up.conv.0.weight|[512, 128, 1, 1]|torch.float32|65536|
|pm.hyper_dec.conv.0.up.conv.0.bias|[512]|torch.float32|512|
|pm.hyper_dec.conv.0.conv.dc.0.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.0.conv.dc.0.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.0.conv.dc.2.weight|[128, 1, 3, 3]|torch.float32|1152|
|pm.hyper_dec.conv.0.conv.dc.2.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.0.conv.dc.3.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.0.conv.dc.3.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.0.conv.ffn.0.weight|[512, 128, 1, 1]|torch.float32|65536|
|pm.hyper_dec.conv.0.conv.ffn.0.bias|[512]|torch.float32|512|
|pm.hyper_dec.conv.0.conv.ffn.2.weight|[128, 256, 1, 1]|torch.float32|32768|
|pm.hyper_dec.conv.0.conv.ffn.2.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.1.up.conv.0.weight|[512, 128, 1, 1]|torch.float32|65536|
|pm.hyper_dec.conv.1.up.conv.0.bias|[512]|torch.float32|512|
|pm.hyper_dec.conv.1.conv.dc.0.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.1.conv.dc.0.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.1.conv.dc.2.weight|[128, 1, 3, 3]|torch.float32|1152|
|pm.hyper_dec.conv.1.conv.dc.2.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.1.conv.dc.3.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.1.conv.dc.3.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.1.conv.ffn.0.weight|[512, 128, 1, 1]|torch.float32|65536|
|pm.hyper_dec.conv.1.conv.ffn.0.bias|[512]|torch.float32|512|
|pm.hyper_dec.conv.1.conv.ffn.2.weight|[128, 256, 1, 1]|torch.float32|32768|
|pm.hyper_dec.conv.1.conv.ffn.2.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.2.dc.0.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.2.dc.0.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.2.dc.2.weight|[128, 1, 3, 3]|torch.float32|1152|
|pm.hyper_dec.conv.2.dc.2.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.2.dc.3.weight|[128, 128, 1, 1]|torch.float32|16384|
|pm.hyper_dec.conv.2.dc.3.bias|[128]|torch.float32|128|
|pm.hyper_dec.conv.2.ffn.0.weight|[512, 128, 1, 1]|torch.float32|65536|
|pm.hyper_dec.conv.2.ffn.0.bias|[512]|torch.float32|512|
|pm.hyper_dec.conv.2.ffn.2.weight|[128, 256, 1, 1]|torch.float32|32768|
|pm.hyper_dec.conv.2.ffn.2.bias|[128]|torch.float32|128|
|pm.temporal_prior_encoder.down.weight|[256, 256, 2, 2]|torch.float32|262144|
|pm.temporal_prior_encoder.down.bias|[256]|torch.float32|256|
|pm.temporal_prior_encoder.conv.dc.0.weight|[256, 256, 1, 1]|torch.float32|65536|
|pm.temporal_prior_encoder.conv.dc.0.bias|[256]|torch.float32|256|
|pm.temporal_prior_encoder.conv.dc.2.weight|[256, 1, 3, 3]|torch.float32|2304|
|pm.temporal_prior_encoder.conv.dc.2.bias|[256]|torch.float32|256|
|pm.temporal_prior_encoder.conv.dc.3.weight|[256, 256, 1, 1]|torch.float32|65536|
|pm.temporal_prior_encoder.conv.dc.3.bias|[256]|torch.float32|256|
|pm.temporal_prior_encoder.conv.ffn.0.weight|[1024, 256, 1, 1]|torch.float32|262144|
|pm.temporal_prior_encoder.conv.ffn.0.bias|[1024]|torch.float32|1024|
|pm.temporal_prior_encoder.conv.ffn.2.weight|[256, 512, 1, 1]|torch.float32|131072|
|pm.temporal_prior_encoder.conv.ffn.2.bias|[256]|torch.float32|256|
|pm.y_prior_fusion.conv.0.dc.0.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.0.dc.0.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.0.dc.2.weight|[384, 1, 3, 3]|torch.float32|3456|
|pm.y_prior_fusion.conv.0.dc.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.0.dc.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.0.dc.3.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.0.ffn.0.weight|[1536, 384, 1, 1]|torch.float32|589824|
|pm.y_prior_fusion.conv.0.ffn.0.bias|[1536]|torch.float32|1536|
|pm.y_prior_fusion.conv.0.ffn.2.weight|[384, 768, 1, 1]|torch.float32|294912|
|pm.y_prior_fusion.conv.0.ffn.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.1.dc.0.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.1.dc.0.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.1.dc.2.weight|[384, 1, 3, 3]|torch.float32|3456|
|pm.y_prior_fusion.conv.1.dc.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.1.dc.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.1.dc.3.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.1.ffn.0.weight|[1536, 384, 1, 1]|torch.float32|589824|
|pm.y_prior_fusion.conv.1.ffn.0.bias|[1536]|torch.float32|1536|
|pm.y_prior_fusion.conv.1.ffn.2.weight|[384, 768, 1, 1]|torch.float32|294912|
|pm.y_prior_fusion.conv.1.ffn.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.2.dc.0.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.2.dc.0.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.2.dc.2.weight|[384, 1, 3, 3]|torch.float32|3456|
|pm.y_prior_fusion.conv.2.dc.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.2.dc.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.2.dc.3.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.2.ffn.0.weight|[1536, 384, 1, 1]|torch.float32|589824|
|pm.y_prior_fusion.conv.2.ffn.0.bias|[1536]|torch.float32|1536|
|pm.y_prior_fusion.conv.2.ffn.2.weight|[384, 768, 1, 1]|torch.float32|294912|
|pm.y_prior_fusion.conv.2.ffn.2.bias|[384]|torch.float32|384|
|pm.y_prior_fusion.conv.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_prior_fusion.conv.3.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.0.adaptor.weight|[384, 512, 1, 1]|torch.float32|196608|
|pm.y_spatial_prior.conv.0.adaptor.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.0.dc.0.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_spatial_prior.conv.0.dc.0.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.0.dc.2.weight|[384, 1, 3, 3]|torch.float32|3456|
|pm.y_spatial_prior.conv.0.dc.2.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.0.dc.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_spatial_prior.conv.0.dc.3.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.0.ffn.0.weight|[1536, 384, 1, 1]|torch.float32|589824|
|pm.y_spatial_prior.conv.0.ffn.0.bias|[1536]|torch.float32|1536|
|pm.y_spatial_prior.conv.0.ffn.2.weight|[384, 768, 1, 1]|torch.float32|294912|
|pm.y_spatial_prior.conv.0.ffn.2.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.1.dc.0.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_spatial_prior.conv.1.dc.0.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.1.dc.2.weight|[384, 1, 3, 3]|torch.float32|3456|
|pm.y_spatial_prior.conv.1.dc.2.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.1.dc.3.weight|[384, 384, 1, 1]|torch.float32|147456|
|pm.y_spatial_prior.conv.1.dc.3.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.1.ffn.0.weight|[1536, 384, 1, 1]|torch.float32|589824|
|pm.y_spatial_prior.conv.1.ffn.0.bias|[1536]|torch.float32|1536|
|pm.y_spatial_prior.conv.1.ffn.2.weight|[384, 768, 1, 1]|torch.float32|294912|
|pm.y_spatial_prior.conv.1.ffn.2.bias|[384]|torch.float32|384|
|pm.y_spatial_prior.conv.2.weight|[256, 384, 1, 1]|torch.float32|98304|
|pm.y_spatial_prior.conv.2.bias|[256]|torch.float32|256|
