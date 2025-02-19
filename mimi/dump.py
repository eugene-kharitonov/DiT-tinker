from moshi.models import loaders
import torchaudio
import librosa
import torch
import numpy as np
import os



SPLIT = 'dev-clean'
BSZ = 16


class RandomSliceDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, seconds_to_extract):
    self.sr = 16_000.0

    self.dataset = dataset
    self.seconds_to_extract = seconds_to_extract

    filtered_indexes = []

    for i in range(len(dataset)):
      example = dataset[i]
      audio, *_ = example

      # first dim is const=1
      length = audio.shape[1] / self.sr
      if length >= self.seconds_to_extract:
        filtered_indexes.append(i)
    
    print(f'After filtering, have {len(filtered_indexes)} out of {len(dataset)}')
    self.filtered_indexes = filtered_indexes

  def __len__(self):
    return len(self.filtered_indexes)

  def __getitem__(self, n: int):
    example = self.dataset[self.filtered_indexes[n]]
    audio, *rest = example
    frames = audio.shape[1]
    target_length = int(self.sr * self.seconds_to_extract)
    high = frames - target_length
    assert high >= 0.0

    # TODO: check boundaries

    start = np.random.uniform(
        low=0,
        high=high
    )
    start = int(start)

    audio = audio[:, start:start + target_length]
    audio = librosa.resample(audio.numpy(), orig_sr=self.sr, target_sr=24_000)
    return torch.from_numpy(audio)

@torch.no_grad
def embed_audio(x, quantize=False):
  emb = mimi._encode_to_unquantized_latent(x)
  if quantize:
    codes = mimi.quantizer.encode(emb)
    emb = mimi.quantizer.decode(codes)
  else:
    # Should I tinker with the projected or pre-projected embeddings, actually?
    emb1 = mimi.quantizer.rvq_first.input_proj(emb)
    emb1 =  mimi.quantizer.rvq_first.output_proj(emb1)

    emb2 = mimi.quantizer.rvq_rest.input_proj(emb)
    emb2 = mimi.quantizer.rvq_rest.output_proj(emb2)

    emb = emb1 + emb2
  return emb

import time
import numpy as np

def dump(root, loader):
  start = time.time()
  ex_id = 0
  for i, b in enumerate(loader):
    b = b.cuda()
    emb = embed_audio(b)
    emb = emb.cpu().numpy()
    
    for j in range(emb.shape[0]):
      fname = f'{ex_id}.npy'
      np.save(fname, emb[j])
      ex_id += 1
    
    if i % 10 == 0 and i > 0:
      print(f'Batch {i}, mean time per batch {(time.time() - start) / i}')
      print(emb.shape)
  print(f'Total time {(time.time() - start) / 60} minutes')


def main():
	hf_repo = 'kyutai/moshika-pytorch-bf16'
	path = loaders.hf_hub_download(hf_repo, loaders.MIMI_NAME)

	mimi = loaders.get_mimi(path, 'cuda').eval()

	dataset = torchaudio.datasets.LIBRISPEECH(download=True, root='.', url=SPLIT)
	sliced = RandomSliceDataset(dataset, seconds_to_extract=5.04)
	loader = torch.utils.data.DataLoader(sliced, batch_size=BSZ, shuffle=False, collate_fn=torch.stack)

	os.mkdir(SPLIT)

	dump(SPLIT, loader)
	


if __name__ == '__main__':
	main()

