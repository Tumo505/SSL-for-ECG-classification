"""Check finite gradients and checkpoint restoration on actual PTB-XL signals."""
from pathlib import Path
import json
import torch
from ssrl_ecg.data.ptbxl import DEFAULT_DATA_ROOT,load_ptbxl_metadata,PTBXLRecordDataset
from ssrl_ecg.models.cnn import ECGEncoder1DCNN,ECGClassifier,SSLReconstructionModel
from ssrl_ecg.models.improved_cnn import ImprovedECGEncoder,ImprovedECGClassifier
from ssrl_ecg.models.cv_improvements import CVRiskPredictor,MultiScaleECGAnalysis
from ssrl_ecg.models.resnet1d import ResNet1D
from ssrl_ecg.train_ssl_contrastive import SSLContrastiveModel
from ssrl_ecg.models.simclr import NTXentLoss
from ssrl_ecg.experiment import cpu_state,write_json

torch.set_num_threads(4)
torch.manual_seed(42)
df,labels=load_ptbxl_metadata(DEFAULT_DATA_ROOT)
indices=df.index[(df.strat_fold<=8)&(labels.sum(1)>0)].to_numpy()[:8]
ds=PTBXLRecordDataset(DEFAULT_DATA_ROOT,df,labels,indices)
x=torch.stack([ds[i][0] for i in range(len(ds))])
y=torch.stack([ds[i][1] for i in range(len(ds))])
models={
 'cnn':ECGClassifier(ECGEncoder1DCNN()),
 'improved_cnn':ImprovedECGClassifier(ImprovedECGEncoder()),
 'cv_predictor':CVRiskPredictor(ECGEncoder1DCNN()),
 'resnet':ResNet1D(),
 'reconstruction':SSLReconstructionModel(),
 'legacy_contrastive_model':SSLContrastiveModel(),
 'multiscale':MultiScaleECGAnalysis(),
}
results={}
for name,model in models.items():
 optimizer=torch.optim.Adam(model.parameters(),lr=.001)
 before=cpu_state(model)
 model.train()
 output=model(x)
 if name=='reconstruction':
  loss=torch.nn.functional.mse_loss(output,x)
 elif name=='legacy_contrastive_model':
  second=model(x+.03*torch.randn_like(x))
  loss=NTXentLoss(.07)(output,second)
 elif name=='multiscale':
  loss=output.square().mean()
 else:
  assert output.shape==y.shape
  loss=torch.nn.functional.binary_cross_entropy_with_logits(output,y)
 assert torch.isfinite(loss)
 optimizer.zero_grad(set_to_none=True)
 loss.backward()
 gradients=[p.grad for p in model.parameters() if p.grad is not None]
 assert gradients and all(torch.isfinite(g).all() for g in gradients)
 optimizer.step()
 assert any(not torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items())
 model.eval()
 with torch.no_grad(): expected=model(x)
 state=cpu_state(model)
 model.load_state_dict(state)
 with torch.no_grad(): restored=model(x)
 torch.testing.assert_close(expected,restored,rtol=0,atol=0)
 results[name]=dict(loss=float(loss.detach()),output_shape=list(output.shape),finite_gradients=True,checkpoint_restore_exact=True)
 print(name,'passed',flush=True)
write_json(Path('results/pipeline_validation/model_validation.json'),dict(status='passed',data='PTB-XL training records',device='cpu',checks=results))
