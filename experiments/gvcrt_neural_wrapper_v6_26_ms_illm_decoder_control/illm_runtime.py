"""Selective official MS-ILLM imports and frozen local-label supervision."""
from io21 import *
import types,importlib,torch
import torch.nn.functional as F
OFFICIAL=ROOT/'vendor/NeuralCompression-3f122808255a2278eea049119367eb3abc434862'
WEIGHT=ROOT/'weights/vqvae_xcit_p8_ch64_cb1024_h8.ckpt'
def official_modules():
 base=OFFICIAL/'neuralcompression'
 for sub in ('','layers','models','zoo','_outputs'):
  name='neuralcompression'+('.'+sub if sub else '')
  if name not in sys.modules:
   m=types.ModuleType(name);m.__path__=[str(base/sub)];sys.modules[name]=m
 def get(sub,name):return importlib.import_module('neuralcompression.'+sub+'.'+name)
 output=get('_outputs','_vqvae_autoencoder_output').VqVaeAutoencoderOutput;sys.modules['neuralcompression'].VqVaeAutoencoderOutput=output
 layers=sys.modules['neuralcompression.layers']
 for file,klass in (('_channel_norm','ChannelNorm2D'),('_xca','XCA'),('_normalize_latent','NormalizeLatent'),('_project_latent','ProjectLatent'),('_vq_bottleneck','VQBottleneck')):setattr(layers,klass,get('layers',file).__dict__[klass])
 # Official VQ forward updates hit counters even in eval. Omit only that
 # bookkeeping in the local eval adapter; quantization and index semantics unchanged.
 cls=layers.VQBottleneck
 if not getattr(cls,'_eval_buffer_guard',False):
  original=cls.forward
  def guarded(self,output):
   if self.training:return original(self,output)
   from dataclasses import replace
   z=self.quantize_vectors(preq_latents=output.latent,codebook=self.codebooks[0],probabilistic=False)
   return replace(output,latent=z.latent,prequantized_latent=z.prequantized_latent,commitment_loss=z.commitment_loss,embedding_loss=z.embedding_loss,num_bytes=int(output.latent.shape[0]*output.latent.shape[-2]*output.latent.shape[-1]*10/8),quantize_distances=z.distances,indices=z.indices)
  cls.forward=guarded;cls._eval_buffer_guard=True
 models=sys.modules['neuralcompression.models'];models.VqVaeXCiTAutoencoder=get('models','_vqvae_xcit_autoencoder').VqVaeXCiTAutoencoder
 return get('zoo','_msillm_vqvae'),get('models','_vqvae_projector'),get('models','_unet_discriminator')
def labeler(device):
 zoo,proj,_=official_modules();model=zoo.vqvae_xcit_p8_ch64_cb1024_h8(pretrained=False);s=torch.load(WEIGHT,map_location='cpu',weights_only=True);model.load_state_dict(s,strict=True);assert tensor_hash(model.state_dict())==tensor_hash(s)
 m=proj.VqVaeProjector(model).to(device).eval().requires_grad_(False)
 dump(ROOT/'audits/illm_source.json',dict(status='PASS',commit='3f122808255a2278eea049119367eb3abc434862',source='https://github.com/facebookresearch/NeuralCompression',weight_source='https://dl.fbaipublicfiles.com/NeuralCompression/2023-muckley-msillm/vqvae_xcit_p8_ch64_cb1024_h8.ckpt',weight_path=str(WEIGHT),weight_bytes=WEIGHT.stat().st_size,weight_sha256=sha(WEIGHT),module_hash=tensor_hash(m.state_dict()),strict_full_autoencoder_load=True,adaptations=['selective package imports, no environment installation','omit eval VQ codebook_hits bookkeeping, identical quantize_vectors','ConditionalUnet context_in 220 to 1024, context_out16, factor8,1025 classes, identity norm'],source_files={str(p):sha(p) for p in OFFICIAL.rglob('*.py')},torch=torch.__version__))
 return m
def local_ce(scores,target):
 return -F.log_softmax(scores,dim=1).gather(1,target[:,None]).mean()
class ILLM(torch.nn.Module):
 def __init__(self,feature,disc):super().__init__();self.feature=feature;self.discriminator=disc;self._target=None;self._labels={};self.weight_sha256=sha(WEIGHT)
 def d_parameters(self):return list(self.discriminator.parameters())
 def d_hash(self):return tensor_hash(self.discriminator.state_dict())
 def labels(self,gt):
  key=hashlib.sha256(gt.detach().cpu().contiguous().numpy().tobytes()).hexdigest()+self.weight_sha256
  p=ROOT/'label_cache'/f'{key}.pt'
  if key not in self._labels:
   if p.exists():y=torch.load(p,map_location=gt.device,weights_only=True)
   else:
    with torch.no_grad():
     before=tensor_hash(dict(self.feature.named_buffers()));raw=self.feature(gt);assert raw.numel()==gt.shape[0]*32*32;y=raw.reshape(gt.shape[0],32,32).long();assert y.min()>=0 and y.max()<1024
     assert before==tensor_hash(dict(self.feature.named_buffers()));atomic_torch(p,y.cpu())
   self._labels={key:y}
  return self._labels[key]
 def set_target(self,gt):self._target=self.labels(gt).detach()
 def scores(self,x,c):assert not c.requires_grad;return [self.discriminator(x,c.detach().float())]
 def adversarial(self,x,c):
  s=self.scores(x,c)[0];assert s.shape==torch.Size([1,1025,32,32]);return local_ce(s,self._target+1)
 def d_loss(self,gt,fake,c):
  y=self.labels(gt);return .5*local_ce(self.scores(gt,c)[0],y+1)+.5*local_ce(self.scores(fake.detach(),c)[0],torch.zeros_like(y))
 def g_mode(self):self.eval();self.feature.eval().requires_grad_(False);self.discriminator.requires_grad_(False)
 def d_mode(self):self.discriminator.train().requires_grad_(True);self.feature.eval().requires_grad_(False)
def create_illm(device):
 _,_,unet=official_modules()
 with torch.random.fork_rng(devices=[device.index or 0]):
  torch.manual_seed(20261026);torch.cuda.manual_seed_all(20261026);f=labeler(device);d=unet.ConditionalUnetDiscriminator(output_channels=1025,output_downsampling_factor=8,context_in=1024,context_out=16,norm_type='identity').to(device)
 return ILLM(f,d).to(device)
def create_for(branch,device):
 if branch=='C_decoder_illm':return create_illm(device)
 from projected import create
 return create(device)
