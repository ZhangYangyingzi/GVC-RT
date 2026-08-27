import os

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8, bias_pixel_shuffle_8

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368

block_in = 512
block_out = block_in * 4
z_channels = 18
hidden_dim = g_ch_enc_dec * 4
g_ch_recon = 320

from .improved_model_gvcrt import Decoder as PretrainedDecoder
from collections import OrderedDict

class PretrainedReconWrapper(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.decoder = PretrainedDecoder(
            ch=128,
            out_ch=3,
            in_channels=18,
            num_res_blocks=2,
            z_channels=18,
            ch_mult=(1, 1, 2, 2, 4),
            resolution=256
        )

    def load_from_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when loading the reconstruction checkpoint")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        new_sd = OrderedDict()
        for k, v in sd.items():
            key = k.replace("module.", "")
            if key.startswith("decoder."):
                key = key.replace("decoder.", "")

            new_sd[key] = v

        missing, unexpected = self.decoder.load_state_dict(new_sd, strict=False)
        print(f"[ReconWrapper] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def forward(self, x, quant_step=None):
        return self.decoder(x, quant_step)

class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(out, quant_step)

        return self.forward_cuda(out, quant_step)

    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)

    def forward_cuda(self, out, quant_step):
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)

class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.conv_in = DepthConvBlock(N, block_in)
        self.dec_1 = nn.Sequential(
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
            DepthConvBlock(block_in, block_in),
            nn.GroupNorm(32, block_in, eps=1e-6),
        )
        self.conv_out = DepthConvBlock(block_in, z_channels)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)
        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv_in(x)
        out = out * quant_step
        out = self.dec_1(out)
        out = out * torch.sigmoid(out)
        out = self.conv_out(out)
        return torch.clamp(out, -1., 1.)

    def forward_cuda(self, x, quant_step):
        out = self.conv_in(x, quant_step=quant_step)
        out = self.dec_1[0](out)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out)
        out = self.dec_1[13](out)
        out = self.dec_1[14](out)
        out = self.dec_1[15](out)
        out = out * torch.sigmoid(out)
        out = self.conv_out(out)
        return torch.clamp(out, -1., 1.)

class DMCI(CompressionModel):
    def __init__(self, N=256, z_channel=128, encoder_ckpt_path=None):
        super().__init__(z_channel=z_channel)

        self.enc = IntraEncoder(N)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N)
        self.recon_generation_net = PretrainedReconWrapper()

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), block_in, 1, 1)))
        self.q_scale_recon = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_recon, 1, 1)))

        if encoder_ckpt_path is not None:
            print(f"[DMCI] Loading pretrained Encoder from {encoder_ckpt_path}...")
            self.init_from_ckpt(encoder_ckpt_path)

    def init_from_ckpt(self, path):
        if path is None or not os.path.exists(path):
            print(f"[DMCI] Warning: Checkpoint path {path} does not exist. Skipping.")
            return

        print(f"==> [DMCI] Attempting full model load from: {path}")
        checkpoint = torch.load(path, map_location="cpu")

        if 'student' in checkpoint:
            sd = checkpoint['student']
            print("    🌟 [SUCCESS] Found 'ema' key! Extracting EMA weights for testing...")
        elif 'ema_shadow' in checkpoint:
            sd = checkpoint['ema_shadow']
            print("    🌟 [SUCCESS] Found 'ema_shadow' key! Extracting EMA weights for testing...")
        elif 'student' in checkpoint:
            sd = checkpoint['student']
            print("    ⚠️ [WARNING] 'ema' key NOT found! Falling back to raw 'student' weights...")
        else:
            sd = checkpoint.get("state_dict", checkpoint)
            print("    ⚠️ [WARNING] No 'ema' or 'student' key. Using default state_dict...")

        model_state_dict = self.state_dict()
        new_sd = OrderedDict()

        load_stats = {
            "enc (Encoder)": 0,
            "dec (Latent Dec)": 0,
            "recon (Recon Head)": 0,
            "entropy (Prior/Hyper)": 0
        }

        for k, v in sd.items():
            clean_k = k.replace("module.", "")

            if clean_k in model_state_dict:
                if v.shape == model_state_dict[clean_k].shape:
                    new_sd[clean_k] = v
                    if clean_k.startswith("enc."):
                        load_stats["enc (Encoder)"] += 1
                    elif clean_k.startswith("dec."):
                        load_stats["dec (Latent Dec)"] += 1
                    elif "recon_generation_net" in clean_k:
                        load_stats["recon (Recon Head)"] += 1
                    else:
                        load_stats["entropy (Prior/Hyper)"] += 1
                else:
                    print(f"    [Shape Mismatch] Skip {clean_k}: {v.shape} vs {model_state_dict[clean_k].shape}")
            else:
                pass

        msg = self.load_state_dict(new_sd, strict=False)

        print("-" * 55)
        print(f"{'Sub-Module':<25} | {'Parameters Loaded'}")
        print("-" * 55)
        for mod, count in load_stats.items():
            status = "✅" if count > 0 else "❌ FAILED"
            print(f"{mod:<25} | {count:<10} {status}")
        print("-" * 55)

        if len(msg.missing_keys) > 0:
            print(f"    Notice: Missing {len(msg.missing_keys)} keys (e.g. {msg.missing_keys[:3]}...)")

        print(f"==> [DMCI] Full Load Finished.")

    def forward(self, x, qp):
        device = x.device
        curr_q_enc = self.q_scale_enc[qp:qp + 1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp + 1, :, :, :]
        curr_q_recon = self.q_scale_recon[qp:qp + 1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
            self.compress_prior_4x(
                y, params, self.y_spatial_prior_reduction,
                self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        cuda_event = torch.cuda.Event()
        cuda_event.record()

        codeword_soft = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(codeword_soft, curr_q_recon)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
            self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def compress(self, x, qp):
        device = x.device
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        curr_q_recon = self.q_scale_recon[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
            self.compress_prior_4x(
                y, params, self.y_spatial_prior_reduction,
                self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        cuda_event = torch.cuda.Event()
        cuda_event.record()

        codeword_soft = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(codeword_soft, curr_q_recon)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
            self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        curr_q_recon = self.q_scale_recon[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, dtype)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(params, self.y_spatial_prior_reduction,
                                         self.y_spatial_prior_adaptor_1,
                                         self.y_spatial_prior_adaptor_2,
                                         self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        codeword_soft = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(codeword_soft, curr_q_recon)
        return {"x_hat": x_hat}
