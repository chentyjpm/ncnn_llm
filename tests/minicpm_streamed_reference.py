"""Memory-bounded *official-model test oracle* tensor adapters.

Only embedding lookup and the final bias-free linear projection are streamed.
Decoder, visual-token scatter, M-RoPE and caches remain official Transformers.
This is not an ncnn runtime or alternate inference engine.
"""
from pathlib import Path

import torch
from safetensors import safe_open


def locate_tensor(model_dir, key):
    matches = []
    for path in sorted(Path(model_dir).glob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as checkpoint:
            if key in checkpoint.keys():
                matches.append((str(path), tuple(checkpoint.get_slice(key).get_shape())))
    if len(matches) != 1:
        raise ValueError(f'expected one checkpoint tensor {key}, found {len(matches)}')
    path, shape = matches[0]
    if len(shape) != 2:
        raise ValueError(f'expected a matrix for {key}, found {shape}')
    return path, shape


class StreamedEmbedding(torch.nn.Module):
    """Exact selected rows of checkpoint embeddings, promoted to CPU FP32."""
    def __init__(self, model_dir, key='model.language_model.embed_tokens.weight'):
        super().__init__()
        self.path, shape = locate_tensor(model_dir, key)
        self.key = key
        self.num_embeddings, self.embedding_dim = shape
        self.padding_idx = None  # Preserve every checkpoint row, including PAD.

    @torch.no_grad()
    def forward(self, input_ids):
        if input_ids.device.type != 'cpu' or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('streamed reference embedding requires CPU integer IDs')
        flat = input_ids.reshape(-1)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings):
            raise IndexError('token ID outside checkpoint vocabulary')
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        selected = torch.empty((unique.numel(), self.embedding_dim), dtype=torch.float32)
        with safe_open(self.path, framework='pt', device='cpu') as checkpoint:
            source = checkpoint.get_slice(self.key)
            for i, token in enumerate(unique.tolist()):
                selected[i].copy_(source[token:token+1, :].reshape(-1).float())
        return selected[inverse].reshape(*input_ids.shape, self.embedding_dim)


class StreamedLMHead(torch.nn.Module):
    """Same FP32 linear map as nn.Linear; vocabulary rows bounded by chunk_rows.

    Full logits are retained (under 1 MiB for one checkpoint-vocabulary token),
    so greedy argmax and first-prefill-logit comparisons use every vocabulary ID.
    """
    def __init__(self, model_dir, chunk_rows=16384, key='lm_head.weight'):
        super().__init__()
        if chunk_rows < 1:
            raise ValueError('head chunk_rows must be positive')
        self.path, shape = locate_tensor(model_dir, key)
        self.out_features, self.in_features = shape
        self.key = key
        self.chunk_rows = chunk_rows

    @torch.no_grad()
    def forward(self, hidden):
        if hidden.device.type != 'cpu' or hidden.dtype != torch.float32:
            raise ValueError('streamed reference head requires CPU FP32 hidden states')
        if hidden.shape[-1] != self.in_features:
            raise ValueError('hidden width does not match checkpoint LM head')
        result = torch.empty((*hidden.shape[:-1], self.out_features), dtype=torch.float32)
        with safe_open(self.path, framework='pt', device='cpu') as checkpoint:
            source = checkpoint.get_slice(self.key)
            for first in range(0, self.out_features, self.chunk_rows):
                end = min(first+self.chunk_rows, self.out_features)
                weight = source[first:end, :].float()
                result[..., first:end] = torch.nn.functional.linear(hidden, weight)
                del weight
        return result


def bind_streamed_tables(model, model_dir, chunk_rows=16384):
    """Bind tables into an official MiniCPMV4_7ForConditionalGeneration wrapper."""
    embedding = StreamedEmbedding(model_dir)
    head = StreamedLMHead(model_dir, chunk_rows)
    cfg = model.config.text_config
    if (embedding.num_embeddings, embedding.embedding_dim) != (cfg.vocab_size, cfg.hidden_size):
        raise ValueError('embedding matrix differs from text config')
    if (head.out_features, head.in_features) != (cfg.vocab_size, cfg.hidden_size):
        raise ValueError('LM-head matrix differs from text config')
    model.model.language_model.embed_tokens = embedding
    model.lm_head = head


def self_test():
    """Tiny BF16 checkpoint, official hybrid decoder, scatter, cache and greedy parity."""
    import sys
    import tempfile
    from safetensors.torch import save_file
    from transformers import MiniCPMV4_7Config
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'export'))
    from minicpm_v47_export import LanguageOnly, load_language_only
    torch.set_num_threads(1)
    torch.manual_seed(47)
    cfg = MiniCPMV4_7Config(
        image_token_id=20, image_start_id=21, image_end_id=22,
        slice_start_id=23, slice_end_id=24, newline_id=25,
        text_config={'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
        'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':32,
        'vocab_size':129,'linear_num_key_heads':2,'linear_num_value_heads':2,
        'linear_key_head_dim':16,'linear_value_head_dim':16,'linear_conv_kernel_dim':4,
        'layer_types':['linear_attention','full_attention'],'partial_rotary_factor':1.0,
        'rope_parameters':{'rope_type':'default','rope_theta':10000000.0,
                           'partial_rotary_factor':1.0,'mrope_section':[6,5,5]},
        'tie_word_embeddings':False})
    cfg._attn_implementation='eager'; cfg.text_config._attn_implementation='eager'
    initial = LanguageOnly(cfg).eval()
    with tempfile.TemporaryDirectory() as td:
        state = {k:v.to(torch.bfloat16).contiguous() for k,v in initial.state_dict().items()}
        save_file(state, str(Path(td)/'model.safetensors'))
        del initial,state
        wrappers=[]
        for streaming in (False,True):
            with torch.device('meta'):
                model=MiniCPMV4_7ForConditionalGeneration(cfg)
            loaded=load_language_only(td,cfg,component='decoder' if streaming else 'all')
            model.model.language_model=loaded.model.language_model
            model.lm_head=loaded.lm_head
            del loaded
            if streaming: bind_streamed_tables(model,td,chunk_rows=17)
            wrappers.append(model.eval())
        dense,streamed=wrappers
        ids=torch.tensor([[0,5,5,128,20,0]],dtype=torch.long)
        torch.testing.assert_close(dense.model.language_model.embed_tokens(ids),
                                   streamed.model.language_model.embed_tokens(ids),rtol=0,atol=0)
        for shape in ((1,1,64),(2,3,64)):
            hidden=torch.randn(shape)
            torch.testing.assert_close(streamed.lm_head(hidden),dense.lm_head(hidden),atol=2e-6,rtol=2e-5)
        prompt=torch.tensor([[3,21,20,20,20,20,22,4]])
        mm=(prompt==20).int()
        visual=torch.randn(4,64)
        outputs=[]
        with torch.no_grad():
            for model in wrappers:
                outputs.append(model(input_ids=prompt,mm_token_type_ids=mm,target_sizes=torch.tensor([[8,8]]),
                     mm_encoder_outputs={'image':BaseModelOutputWithPooling(pooler_output=(visual,))},
                     use_cache=True,logits_to_keep=1))
            max_error=0.0
            for step in range(6):
                a,b=outputs
                max_error=max(max_error,float((a.logits-b.logits).abs().max()))
                torch.testing.assert_close(a.logits,b.logits,atol=3e-6,rtol=2e-5)
                token_a=a.logits[:,-1].argmax(-1,keepdim=True)
                token_b=b.logits[:,-1].argmax(-1,keepdim=True)
                torch.testing.assert_close(token_a,token_b,rtol=0,atol=0)
                if step<5:
                    outputs=[m(input_ids=token_a,past_key_values=o.past_key_values,use_cache=True,logits_to_keep=1)
                             for m,o in zip(wrappers,outputs)]
        assert not list(streamed.model.language_model.embed_tokens.parameters())
        assert not list(streamed.lm_head.parameters())
        print(f'PASS: tiny BF16 streamed embedding/head and official hybrid VQA prefill +5 cached decode steps; max logit abs error={max_error:.9g}')
        print('Synthetic parity only; no real model weights loaded.')


if __name__=='__main__':
    self_test()
