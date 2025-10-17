import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from peft import get_peft_model, RandLoraConfig  # Add this importsss

def create_backbone(backbone_name, pretrained=True, checkpoint_path=None):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained, checkpoint_path=checkpoint_path,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim


class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()
        print("DEBUG: Using RandLoRA with rank 32")
        
        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100', pretrained=True)
        
        # Freeze the backbone encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Original expression layers (will be wrapped with RandLoRA)
        self.expression_layers = nn.Sequential(
            nn.Linear(feature_dim, n_exp*2 + 3)
        )
        
        self.n_exp = n_exp
        self.init_weights()
        
        # Apply RandLoRA to expression_layers
        self._apply_randlora()

    def _apply_randlora(self):
        """Apply RandLoRA to the expression layers"""
        config = RandLoraConfig(
            r=32,  # rank - you can tune this
            target_modules=["0"],  # target the first (and only) linear layer in Sequential
            # randlora_alpha=640,  # typically 20 * r
            # randlora_dropout=0.0,
            # bias="none",
            # task_type="FEATURE_EXTRACTION"  # since this is not a standard transformers model
        )
        
        # Wrap expression_layers with RandLoRA
        self.expression_layers = get_peft_model(self.expression_layers, config)

    def init_weights(self):
        # Initialize before applying RandLoRA
        if hasattr(self.expression_layers, 'module'):
            self.expression_layers.module[0].weight.data *= 0.1
            self.expression_layers.module[0].bias.data *= 0.1
        else:
            self.expression_layers[0].weight.data *= 0.1
            self.expression_layers[0].bias.data *= 0.1

    def forward(self, img):
        features = self.encoder(img)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        parameters = self.expression_layers(features).reshape(img.size(0), -1)
        
        outputs = {}
        outputs['expression_params'] = parameters[..., :self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[..., self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[..., self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[..., self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)
        outputs["image_feature"] = features
        return outputs
    
class FlameParamsNetSmirk(nn.Module):
    def __init__(self, exp_dim=50):
        super(FlameParamsNetSmirk, self).__init__()
        
        
        self.model_path="./assets/smirk/pretrained_models/SMIRK_em1.pt"
        self.expression_encoder = ExpressionEncoder(n_exp=exp_dim)
        self.exp_dim=exp_dim
        self.load_initial_state() 

    
    def forward(self, img):
        return self.expression_encoder(img)
        
    def load_initial_state(self):
        checkpoint = torch.load(self.model_path)

        # Load only encoder (backbone) weights
        checkpoint_expression = {
            k.replace('smirk_encoder.expression_encoder.', ''): v
            for k, v in checkpoint.items()
            if 'smirk_encoder.expression_encoder.encoder' in k
        }
        checkpoint_expression_encoder = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint_expression.items()
            if 'encoder' in k
        }
        self.expression_encoder.encoder.load_state_dict(checkpoint_expression_encoder)

    def reload(self,state=0,ckpt_path=None):
        if state==0:
            self._state_dict=self.expression_encoder.state_dict()
            if ckpt_path is not None:
                ckpt_state=torch.load(ckpt_path)
                self.load_state_dict(ckpt_state)
            else:        
                self.load_initial_state()
        else:
            self.expression_encoder.load_state_dict(self._state_dict)
