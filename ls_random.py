#! /usr/bin/env python

import manage_gpus as mgp

from pysndfile import sndio
import os
from argparse import ArgumentParser


def process_text_to_audio(out_dir, text_path, num_variants, verbose=True):
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


    if verbose:
        print("retrieving lina-speech checkpoint path ")
    lina_ckpt = hf_hub_download(repo_id="lina-speech/all-models", filename="lina_gla_gigaspeech_d1024l12_convblind_shortconv_lr2e-4/last.ckpt")
    if verbose:
        print("retrieving WavTokenizer checkpoint path")
    wt_ckpt = hf_hub_download(repo_id="novateur/WavTokenizer-medium-speech-75token", filename="wavtokenizer_medium_speech_320_24k.ckpt")
    wt_config = hf_hub_download(repo_id="novateur/WavTokenizer-medium-speech-75token", filename="wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")

    if verbose:
        print("instantiate models")
    model = TrainLina.load_from_checkpoint(lina_ckpt).model.eval()
    if verbose:
        print("Done Lina-Speech")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe256.json")

    wavtokenizer = WavTokenizer.from_pretrained0802(wt_config, wt_ckpt).to("cuda")
    if verbose:
        print("Done WavTokenizer")
    bandwidth_id = torch.tensor([0]).cuda()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    batch_size = num_variants
    with open(text_path, "r") as ft:
        for itext, intxt in enumerate(ft.readlines()):
            txt = intxt.replace("\n", " ")
            txt = "[BOS]" + txt + "[EOS]"
            txt = torch.LongTensor(tokenizer.encode(txt))

            if verbose:
                print("generate tokens")
            _, atts, _, cuts = model.generate_batch(
                                        txt,
                                        batch_size=batch_size,
                                        k=100,
                                        max_seqlen=2000,
                                        device="cuda",
)

            with open(f"{out_dir}/ls_txt_{itext}.txt", "w") as fo:
                fo.write(intxt)
            for ii, x in enumerate(cuts):
                if verbose:
                    print(f"decode txt {itext} to audio {ii}")
                features = wavtokenizer.codes_to_features(x[0][..., :].cuda())
                audio = wavtokenizer.decode(features, bandwidth_id=bandwidth_id).numpy(force=True)
                sndio.write(f"{out_dir}/ls_txt_{itext}_var_{ii}.wav", data=audio.T, rate=24000, enc="pcm16", format="wav")



if __name__ == "__main__":
    parser = ArgumentParser(description="Lina-Speech synthesis with random speakers and expressivity")
    parser.add_argument("text_path", type=str,
                        help="Path to the text file containing the text to be synthesized. There will be one synthesis process for each line")
    parser.add_argument("-o", "--out_dir", type=str, default="lina_random_out",
                        help="Directory to save the generated audio files (Def: %(default)s)")
    parser.add_argument("-nv", "--num_variants", type=int, default=4,
                        help="Number of variants (speakers and expressivity) to generate for each text line.  (Def: %(default)s)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Disable verbose output")
    args = parser.parse_args()

    process_text_to_audio(args.out_dir, args.text_path, args.num_variants, verbose=not args.quiet)
