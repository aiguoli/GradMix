from dataclasses import dataclass


@dataclass
class Config:
    model_idx: str = "ViT"
    Pre_Trained_model_path: str = None
    Prompt_state_path: str = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    cls_token_off: bool = False
    pos_embedding_off: bool = False
    att_module: str = "SimAM"
    enable_sam: bool = False
    backbone_PT_off: bool = False

    gpu_idx: int = 0
    dataroot: str = "data/wbc"
    model_path: str = "saved_models"
    draw_root: str = "runs"

    paint: bool = False
    enable_tensorboard: bool = False
    enable_attention_check: bool = False
    enable_visualize_check: bool = False

    PromptTuning: str = None
    Prompt_Token_num: int = 20
    PromptUnFreeze: bool = False
    linearprobing: bool = False

    augmentation_name: str = "HOGMask"
    ratio_strategy: str = None
    patch_strategy: str = None
    loss_drive_threshold: float = 4.0
    fix_position_ratio: float = None
    fix_patch_size: int = None
    patch_size_jump: str = None

    num_classes: int = 0
    edge_size: int = 384
    data_augmentation_mode: int = 0
    batch_size: int = 8
    num_epochs: int = 50
    intake_epochs: int = 0
    lr: float = 5e-6
    lrf: float = 0.2
    opt_name: str = "Adam"
    check_minibatch: int = 600
    num_workers: int = 2
