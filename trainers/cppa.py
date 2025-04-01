# Standard libraries
import os.path as osp
import copy
from collections import OrderedDict
from typing import Optional

# PyTorch and related libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Dassl framework imports
from dassl.data.data_manager import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# CLIP model and tokenizer
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# Initialize tokenizer for text processing
_tokenizer = _Tokenizer()

def get_entropy(input_):
    """Calculate entropy of input tensor."""
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def compute_im_loss(logits):
    """Compute information maximization loss from logits."""
    softmax_out = nn.Softmax(dim=1)(logits)
    entropy_loss = torch.mean(get_entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
    im_loss = entropy_loss - gentropy_loss
    return im_loss

class ProtoLoss(nn.Module):
    """
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, num_classes: int, device: torch.device, nav_t: float = 1, beta: float = 0, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor,is_use_u2t=True) -> torch.Tensor:
        # Update proportions
        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        source_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        target_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*self.prop.squeeze(1)).sum()
        if is_use_u2t:
            loss = source_loss + target_loss
        else:
            loss = source_loss
        return loss

def load_clip_to_cpu(cfg):
    """
    Load CLIP model to CPU with appropriate configuration.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        Loaded CLIP model instance
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
        
    design_details = {"trainer": 'CPPA',
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "cppa_length": cfg.TRAINER.CPPA.N_CTX,
                      "fusing": cfg.TRAINER.CPPA.FUSING,
                      "parameter_sharing": cfg.TRAINER.CPPA.PS}
                      
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class Encoder(nn.Module):
    """
    Dual-stream encoder for processing both visual and textual data through transformer layers.
    
    This encoder extracts features from images and text by passing them through shared 
    transformer blocks while maintaining separate processing paths.
    """
    def __init__(self, clip_model):
        super().__init__()
        # Visual pathway components
        self.conv1_visual = clip_model.conv1_visual
        self.class_embedding_visual = clip_model.class_embedding_visual
        self.positional_embedding_visual = clip_model.positional_embedding_visual
        self.ln_pre_visual = clip_model.ln_pre_visual
        self.ln_post_visual = clip_model.ln_post_visual
        self.proj_visual = clip_model.proj_visual
        
        # Shared transformer blocks for cross-modal interaction
        self.resblocks = clip_model.resblocks
        
        # Textual pathway components
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x_visual, visual_ctx, prompts_text, tokenized_prompts, compound_prompts_depth):
        """
        Process image and text inputs through the encoder.
        
        Args:
            x_visual: Input image tensor
            visual_ctx: Visual context/prompt embeddings
            prompts_text: Text prompt embeddings
            tokenized_prompts: Tokenized text inputs
            compound_prompts_depth: Depth parameter for prompt processing
            
        Returns:
            Tuple of processed visual and textual features
        """
        # Visual preprocessing
        x_visual = self.conv1_visual(x_visual)  # shape = [*, width, grid, grid]
        x_visual = x_visual.reshape(x_visual.shape[0], x_visual.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_visual = x_visual.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_visual = torch.cat(
            [self.class_embedding_visual.to(x_visual.dtype) + torch.zeros(x_visual.shape[0], 1, x_visual.shape[-1], dtype=x_visual.dtype, device=x_visual.device),
             x_visual], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_visual = x_visual + self.positional_embedding_visual.to(x_visual.dtype)

        # Attach visual prompts to the embeddings
        visual_ctx = visual_ctx.expand(x_visual.shape[0], -1, -1)
        x_visual = torch.cat([x_visual, visual_ctx], dim=1)

        # Apply layer normalization
        x_visual = self.ln_pre_visual(x_visual)

        # Adjust dimensions for transformer (NLD -> LND)
        x_visual = x_visual.permute(1, 0, 2)
        
        # Text preprocessing
        x_text = prompts_text + self.positional_embedding.type(self.dtype)
        x_text = x_text.permute(1, 0, 2)  # NLD -> LND
        
        # Combine inputs for transformer processing
        # Counter (4th element) starts at 0 and tracks prompt processing depth
        combined = [x_visual, x_text, compound_prompts_depth, 0]
        outputs = self.resblocks(combined)
        
        # Visual postprocessing
        x_visual = outputs[0]
        x_visual = x_visual.permute(1, 0, 2)  # LND -> NLD
        x_visual = self.ln_post_visual(x_visual[:, 0, :])
        if self.proj_visual is not None:
            x_visual = x_visual @ self.proj_visual.half()

        # Text postprocessing
        x_text = outputs[1]
        x_text = x_text.permute(1, 0, 2)  # LND -> NLD
        x_text = self.ln_final(x_text).type(self.dtype)
        
        # Extract features from the end-of-text token position
        x_text = x_text[torch.arange(x_text.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x_visual, x_text

class MultiModalPromptLearner(nn.Module):
    """
    Multi-modal prompt learning module that learns prompts for both visual and textual pathways.
    
    This module initializes and manages learnable prompt vectors that are injected into
    the visual and textual processing streams of the model.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CPPA.N_CTX
        ctx_init = cfg.TRAINER.CPPA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        # Initialize text context vectors
        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors_text = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization for text prompts
            print("Initializing text prompts with class-specific information")
            ctx_vectors_text = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_text, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print('CPPA design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CPPA context words (tokens): {n_ctx}")
        self.prompts_text = nn.Parameter(ctx_vectors_text)
        
        # Initialize class-specific components
        self.random_init = cfg.random_init
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # Handle class names with different lengths
        self.name_lens = name_lens
        self.min_len = min(self.name_lens)  # Track minimum length for consistency

        if self.min_len > 1:
            print("Original class name lengths:", name_lens)
            classnames = self.revise_classnames(classnames, name_lens, self.min_len)
            name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
            print("Adjusted class name lengths:", name_lens)

        # Initialize visual context vectors
        ctx_vectors_visual = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_visual, std=0.02)
        self.prompts_visual = nn.Parameter(ctx_vectors_visual)  

        # Create tokenized prompts
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Initialize class-specific token handling
        self.eval_only = cfg.eval_only
        self._init_suffix_dict(classnames, clip_model, dtype)
        self._get_token_classes(dtype)

        # Register token prefix buffer (SOS token)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        
        # Register token suffix buffers with appropriate handling for evaluation
        if not self.eval_only:
            self.register_buffer("token_suffix", embedding[:, 1 + self.min_len + n_ctx:, :])  # EOS
            self.register_buffer("token_suffix_test", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + self.min_len + n_ctx:, :])

        # Store configuration
        self.n_cls = n_cls
        print(f"Number of classes: {n_cls}")
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def revise_classnames(self, classnames, name_lens, min_len):
        """
        Adjust class names to ensure consistent minimum length.
        
        Args:
            classnames: List of class names
            name_lens: List of token lengths for each class name
            min_len: Minimum required length
            
        Returns:
            List of adjusted class names
        """
        if min(name_lens) < min_len:
            for i in range(len(classnames)):
                if name_lens[i] < min_len:
                    classnames[i] = ("<|startoftext|> "*(min_len - name_lens[i])) + classnames[i]
        return classnames  

    def _init_suffix_dict(self, classnames, clip_model, dtype):
        """Initialize dictionary of class name token embeddings."""
        self.suffix_classes = {}
        for name in classnames:
            self.suffix_classes[name] = clip_model.token_embedding(clip.tokenize(name)).type(dtype)
    
    def _get_token_classes(self, dtype):
        """Prepare token embeddings for class names."""
        self.token_classes_all = torch.cat([self.suffix_classes[name] for name in self.suffix_classes]).type(dtype)
        
        if not self.eval_only:
            print("Initializing token classes")
            self.token_classes = self.token_classes_all[:, 1:self.min_len+1, :]
            if self.random_init:
                nn.init.normal_(self.token_classes, std=0.02)
            self.token_classes = nn.Parameter(self.token_classes)
            self.fix_token = copy.deepcopy(self.token_classes)
            self.fix_token.requires_grad = False
        else:
            # In evaluation mode, no gradients needed
            self.token_classes = nn.Parameter(self.token_classes_all[:, 1:self.min_len+1, :], requires_grad=False)

    def construct_prompts(self, ctx, prefix, suffix, label=None, classes=None):
        """
        Construct prompt embeddings by combining context, prefix, and suffix tokens.
        
        Args:
            ctx: Context tokens
            prefix: Prefix tokens (SOS)
            suffix: Suffix tokens (EOS)
            label: Optional label for selecting specific tokens
            classes: Optional class tokens to include
            
        Returns:
            Combined prompt embeddings
        """
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        if classes is not None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    classes, # (dim0, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts

    def forward(self):
        """
        Generate prompt embeddings for both visual and textual modalities.
        
        Returns:
            Tuple of (text_prompts, visual_prompts)
        """
        ctx = self.prompts_text

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        classes = self.token_classes
        prompts_text = self.construct_prompts(ctx, prefix, suffix, classes=classes)

        prompts_visual = self.prompts_visual

        return prompts_text, prompts_visual

class CustomCLIP(nn.Module):
    """
    Custom implementation of CLIP model with prompt learning capabilities.
    
    This model integrates learnable prompts for both visual and textual inputs,
    enabling more effective domain adaptation and transfer learning.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.CPPA.PROMPT_DEPTH >= 1, "For CPPA, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.CPPA.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        
        # Initialize prompt learner and encoder components
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.encoder = Encoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, skip_loss=False, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            image: Input image tensor
            label: Optional class labels for training
            skip_loss: Control flag for training return behavior
            return_features: Control flag for returning additional features
            
        Returns:
            During training: Cross entropy loss if not skip_loss
            During inference: Logits or (logits, image_features, text_features) if return_features
        """
        tokenized_prompts = self.tokenized_prompts
        compound_prompts_depth = self.compound_prompts_depth
        logit_scale = self.logit_scale.exp()

        # Get learnable prompts
        prompts_text, prompts_visual = self.prompt_learner()

        # Encode inputs with prompts
        image_features, text_features = self.encoder(
            image.type(self.dtype), 
            prompts_visual, 
            prompts_text, 
            tokenized_prompts, 
            compound_prompts_depth
        )
        
        # Normalize feature vectors
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        logits = logit_scale * image_features @ text_features.t()

        # Different return behavior based on mode and flags
        if self.prompt_learner.training and not skip_loss:
            return F.cross_entropy(logits, label)
        
        if return_features:
            return logits, image_features, text_features

        return logits

@TRAINER_REGISTRY.register()
class CPPA(TrainerXU):
    """
    Trainer implementation for Cross-modal Prompt Parameter Adaptation (CPPA).
    
    This trainer specializes in adapting vision-language models through learnable
    prompt parameters, enabling effective zero/few-shot transfer learning.
    """
    def check_cfg(self, cfg):
        """Verify configuration settings."""
        assert cfg.TRAINER.CPPA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """Initialize model, optimizer and related components."""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CPPA.PREC == "fp32" or cfg.TRAINER.CPPA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # Initialize prototype loss
        self.proto_loss = ProtoLoss(len(classnames), self.device)
        self.proto_loss.train()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                elif 'prompt_attn' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CPPA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes."""
        from torch.utils.data.distributed import DistributedSampler
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x  # source domain training data
        self.train_loader_u = dm.train_loader_u  # target domain training data
        self.val_loader = dm.val_loader  # source domain testing data
        self.test_loader = dm.test_loader # target domain testing data

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def forward_backward(self, batch_x, batch_u):
        """Perform forward and backward passes."""
        image_x, label_x, image_u = self.parse_batch_train(batch_x=batch_x, batch_u=batch_u)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CPPA.PREC
        if prec == "amp":
            print("AMP precision mode is not fully supported in this implementation")
            with autocast():
                loss = model(image_u) 
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            if self.epoch < 25:
                # Source domain training with cross-entropy
                loss = model(image_x, label_x)
            else:
                # Target domain adaptation with information maximization
                logit_u, image_features, text_features = model(image_u, skip_loss=True, return_features=True)
                im_loss = compute_im_loss(logit_u)
                transfer_loss = self.proto_loss(text_features, image_features, True)
                loss = transfer_loss + im_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def before_epoch(self):
        """Setup before each training epoch."""
        current_epoch = self.epoch + 1
        if current_epoch == 1:
            print("+--------------------------------------------------------------------------------+")
            print(f"Epoch {current_epoch}: Training source domain with class and visual prompts.")
            self.model.prompt_learner.prompts_text.requires_grad = False
            print("Text prompt frozen.")
            self.model.prompt_learner.token_classes.requires_grad = True
            print("Class prompts enabled for training.")
            self.model.prompt_learner.prompts_visual.requires_grad = True
            print("Visual prompts enabled for training.")
            self.cfg.TRAIN.COUNT_ITER = "train_x"
            print("Switched to source domain training mode.")
            
            # List trainable parameters
            enabled = set()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"Trainable parameters: {enabled}")
            
            print("Non-attention trainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad and "encoder.resblocks.0.prompt_attn" not in name:
                    print(name)
            print("+--------------------------------------------------------------------------------+")
            return

        if current_epoch == 13:
            print("+--------------------------------------------------------------------------------+")
            print(f"Epoch {current_epoch}: Training source domain with text and visual prompts")
            self.model.prompt_learner.prompts_text.requires_grad = True
            print("Text prompt enabled for training.")
            self.model.prompt_learner.token_classes.requires_grad = False
            print("Class prompts frozen.")
            self.model.prompt_learner.prompts_visual.requires_grad = True
            print("Visual prompts enabled for training.")
            print("Continuing source domain training.")
            
            # List trainable parameters
            enabled = set()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"Trainable parameters: {enabled}")
            
            print("Non-attention trainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad and "encoder.resblocks.0.prompt_attn" not in name:
                    print(name)
            print("+--------------------------------------------------------------------------------+")
            return

        if current_epoch == 26:
            print("+--------------------------------------------------------------------------------+")
            print(f"Epoch {current_epoch}: Switching to target domain adaptation")
            self.model.prompt_learner.prompts_text.requires_grad = True
            print("Text prompt enabled for training.")
            self.model.prompt_learner.token_classes.requires_grad = False
            print("Class prompts frozen.")
            self.model.prompt_learner.prompts_visual.requires_grad = True
            print("Visual prompts enabled for training.")
            self.cfg.TRAIN.COUNT_ITER = "train_u"
            print("Switched to target domain training.")
            print("Initializing adaptive layers for cross-domain transfer...")
            
            enabled = set()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print("Attention module frozen.")
            print(f"Trainable parameters: {enabled}")
            
            print("Non-attention trainable parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad and "encoder.resblocks.0.prompt_attn" not in name:
                    print(name)
            print("+--------------------------------------------------------------------------------+")
            return

    def after_epoch(self):
        """Perform validation after each epoch."""
        if self.val_loader is not None:
            print("----------------------------------------------------------------------------------------------------------------------------------------")
            print("-----------------------------------------------------Evaluating on source domain-----------------------------------------------------")
            self.test(split="val")
            print("----------------------------------------------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        print("-----------------------------------------------------Evaluating on target domain-----------------------------------------------------")
        self.test(split="test")
        print("----------------------------------------------------------------------------------------------------------------------------------------")
        
    def load_model(self, directory, epoch=None):
        """Load model weights from checkpoint."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} from \"{}\" (epoch = {})".format(name, model_path, epoch))
            
            # set strict=False to allow missing keys like 'token_classes'
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """Generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]