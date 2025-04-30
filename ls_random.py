#! /usr/bin/env python

import manage_gpus as mgp
try:
    mgp.get_gpu_lock()
except  mgp.NoGpuManager:
    pass

import torch
from train_lina import TrainLina
from datasets import load_dataset
from decoder.pretrained import WavTokenizer
from transformers import PreTrainedTokenizerFast

from huggingface_hub import hf_hub_download
from pysndfile import sndio


print("retrieving lina-speech checkpoint path ")
lina_ckpt=hf_hub_download(repo_id="lina-speech/all-models", filename="lina_gla_gigaspeech_d1024l12_convblind_shortconv_lr2e-4/last.ckpt")
print("retrieving WavTokenizer checkpoint path")
wt_ckpt=hf_hub_download(repo_id="novateur/WavTokenizer-medium-speech-75token", filename="wavtokenizer_medium_speech_320_24k.ckpt")
wt_config=hf_hub_download(repo_id="novateur/WavTokenizer-medium-speech-75token", filename="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")

print("instantiate models")
model = TrainLina.load_from_checkpoint(lina_ckpt).model.eval()
print("Done Lina-Speech")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe256.json")

wavtokenizer = WavTokenizer.from_pretrained0802(wt_config, wt_ckpt).to("cuda")
print("Done WavTokenizer")
bandwidth_id = torch.tensor([0]).cuda()



txt = "All of the king's horses could drink from these waters at once, at ease and to their full satisfaction, without exhausting them! The water comes from glaciers, ardent as the distant snowfalls, and it is soft as the loam of the plains."
txt = "[BOS]" + txt + "[EOS]"
batch_size = 4
txt = torch.LongTensor(tokenizer.encode(txt))

print("generate tokens")
_, atts, _, cuts = model.generate_batch(
                            txt,
                            batch_size=batch_size,
                            k=100,
                            max_seqlen=2000,
                            device="cuda",
)


for ii, x in enumerate(cuts):
    print(f"decode features to audio [{ii}]")
    features = wavtokenizer.codes_to_features(x[0][..., :].cuda())
    audio= wavtokenizer.decode(features, bandwidth_id=bandwidth_id).numpy(force=True)
    print(audio.shape)
    sndio.write(f"ls_audio_{ii}.wav", data=audio.T, rate=24000, enc="pcm16", format="wav")
    
