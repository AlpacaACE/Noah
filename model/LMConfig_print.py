from transformers import PretrainedConfig
from typing import List


class LMConfig_print(PretrainedConfig):
    model_type = "Noah"

    def __init__(
            self,
            dim: int = 768, # 128, 256, 512, 640, 768,
            n_layers: int = 16, # 8, 8, 8, 8, 16,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6400, # 6400
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            save_steps: int = 1000,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts: bool = True,
            scoring_func='softmax',
            aux_loss_alpha=0.01,
            seq_aux=True,
            norm_topk_prob=True,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.save_steps = save_steps
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)