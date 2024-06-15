import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock



class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel1(nn.Module):

    def __init__(self, vision_type1, project_dim):
        super(VisionModel1, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type1,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}

class VisionModel2(nn.Module):

    def __init__(self, vision_type2, project_dim):
        super(VisionModel2, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type2,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type1, vision_type2, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel1(vision_type1, project_dim)
        # self.bottleneck = VisionModel2(vision_type2, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]
        norm_name = 'BATCH'

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=1,
            norm_name= norm_name,
            res_block=True
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=feature_dim[3],
            out_channels=feature_dim[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=feature_dim[2],
            out_channels=feature_dim[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.encoder8 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=feature_dim[1],
            out_channels=feature_dim[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.encoder16 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=feature_dim[0],
            out_channels=feature_dim[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        # self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.decoder0 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_dim[3],
            out_channels=24,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)        
        image_features, image_project = image_output['feature'], image_output['project']

        # bottleneck_output = self.bottleneck(image)
        # bottleneck_features, bottleneck_project = bottleneck_output['feature'], bottleneck_output['project']

        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            # image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features]

        # if len(bottleneck_features[0].shape) == 4: 
        #     bottleneck_features = bottleneck_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
        #     bottleneck_features = [rearrange(item,'b c h w -> b (h w) c') for item in bottleneck_features]



        enc0 = self.encoder0(image)
        enc1 = self.encoder2(image_features[0])
        enc2 = self.encoder4(image_features[1])
        enc3 = self.encoder8(image_features[2])
        enc4 = self.encoder16(image_features[3])

        enc4 = rearrange(enc4,'b c h w -> b (h w) c')
        enc3 = rearrange(enc3,'b c h w -> b (h w) c')
        enc2 = rearrange(enc2,'b c h w -> b (h w) c')
        enc1 = rearrange(enc1,'b c h w -> b (h w) c')        

        # os32 = image_features[3]
        os16 = self.decoder16(enc4,enc3, text_embeds[-1])
        os8 = self.decoder8(os16,enc2, text_embeds[-1])
        os4 = self.decoder4(os8,enc1, text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        # os1 = self.decoder1(os4)
        os0 = self.decoder0(os4,enc0)

        out = self.out(os0).sigmoid()

        return out
    
