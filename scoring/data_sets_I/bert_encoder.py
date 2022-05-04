from typing import List, Union, Dict, Optional
import torch
import numpy as np
from farm.infer import Inferencer


class TransformerDEModel:
    def __init__(self, model_path="seyonec/ChemBERTa-zinc-base-v1", **kwargs):
        self.model = model_path
        self.embed_title = kwargs.get("embed_title", False)
        use_gpu: bool = kwargs.get("use_gpu", False)
        self.pooling_strategy: str = kwargs.get("pooling_strategy", "reduce_mean")
        model_version = kwargs.get("model_version", None)
        self.emb_extraction_layer: int = kwargs.get("emb_extraction_layer", -1)
        self.model_format = kwargs.get("model_format", "transformers")
        if kwargs.get("model_format", "transformers") == "transformers":
            self.embedding_model = Inferencer.load(
                self.model,
                task_type="embeddings",
                extraction_strategy=self.pooling_strategy,
                extraction_layer=self.emb_extraction_layer,
                gpu=use_gpu,
                batch_size=1,
                max_seq_len=512,
                num_processes=0,
            )
        else:
            raise NotImplementedError

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(
        self, queries: List[str], batch_size: int, **kwargs
    ) -> np.ndarray:
        emb = self.embedding_model.inference_from_dicts(
            dicts=[{"text": t} for t in queries]
        )
        emb = torch.Tensor([(r["vec"]) for r in emb])
        return emb

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int = 1, **kwargs
    ) -> np.ndarray:
        if self.model_format == "transformers":
            emb = self.embedding_model.inference_from_dicts(dicts=[{"text": corpus}])
            emb = np.array([(r["vec"]) for r in emb])
        return emb
