
import torch
from huggingface_hub import hf_hub_download
from whisperspeech.vq_stoks import RQBottleneckTransformer, Tunables

class BNWhisper(RQBottleneckTransformer):
    @classmethod
    def load_model(cls, ref="streich/bn_whisper", load_from_hf=True):
        if load_from_hf:
            repo_id = ref
            local_filename = hf_hub_download(repo_id=repo_id, filename="rvq.model")
        else:
            local_filename = ref

        load_device = "cuda" if torch.cuda.is_available() else "cpu"
        spec = torch.load(local_filename, map_location=load_device) 

        if load_from_hf:
            # Also download our whisper model and update the config
            whmodel = hf_hub_download(repo_id=repo_id, filename="whisper.pt")
            spec["config"]["whisper_model_name"] = whmodel

        vqmodel = cls(**spec['config'], tunables=Tunables(**Tunables.upgrade(spec.get('tunables', {}))))
        vqmodel.load_state_dict(spec['state_dict'])
        vqmodel.eval()
        return vqmodel

def make_model(
    whmodel: str,
    tunables:Tunables=Tunables(),
    dataset:torch.utils.data.Dataset=None
):
    model = BNWhisper(
        codebook_dim=64, vq_codes=2048, q_depth=1, n_head=12, depth=1,
        downsample=2, threshold_ema_dead_code=0, use_cosine_sim=True,
        whisper_model_name=whmodel, tunables=tunables
    )

    return model

def load_model(*args, **kwargs):
    return BNWhisper.load_model(*args, **kwargs)
