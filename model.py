import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners import lora


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel, pool_stride):
        super(DownsampleBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels + 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels + 9, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        conv_out = self.down(x)
        pool_out = self.maxpool(conv_out)
        return conv_out, pool_out


class Projection(nn.Module):
    def __init__(self, hidden_dim, input_dim, out_dim):
        super(Projection, self).__init__()
        self.linear = nn.Linear(hidden_dim * input_dim ** 2, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x



class MedVisExtractor(nn.Module):
    def __init__(self, in_channels, clip_dim, input_dim, hidden_dim=128):
        super(MedVisExtractor, self).__init__()
        features = [hidden_dim]
        kernel_stride = [[2, 2], [2, 1]]
        self.clip_projection = nn.ModuleList()
        self.down_block = nn.ModuleList()
        self.down_block.append(DownsampleBlock(in_channels, features[0], 2, 2))
        for i in range(len(features) - 1):
            if kernel_stride[i][1] == 1:
                input_dim = input_dim // 2
            else:
                input_dim = input_dim - kernel_stride[i][0] + 1
            self.down_block.append(DownsampleBlock(features[i], features[i + 1], kernel_stride[i][0], kernel_stride[i][1]))
            self.clip_projection.append(Projection(features[i], input_dim, clip_dim))
        self.clip_projection.append(Projection(features[-1], input_dim // 2, clip_dim))

    def forward(self, x, clip_visual_encoder):
        skip = []
        skip_connections = []
        for layer in self.down_block:
            conv_out, pool_out = layer(x)
            x = pool_out
            single_skip = pool_out.view(pool_out.size(0), -1)
            skip.append(single_skip)
        for i, layer in enumerate(self.clip_projection):
            single_skip = layer(skip[i])
            skip_connections.append(single_skip)
        down_features = x.half()
        clip_output = clip_visual_encoder(down_features)
        result = clip_output
        for skip_connection in skip_connections:
            result = 0.3 * skip_connection + 0.7 * result
        final_output = result
        return final_output


class SymptomImageTextAlignmentModel(nn.Module):
    def __init__(self, med_vis_extractor, clip_model, clip_text_adapter):
        super(SymptomImageTextAlignmentModel, self).__init__()
        self.med_vis_extractor = med_vis_extractor
        self.clip_model = clip_model
        self.clip_text_adapter = clip_text_adapter

    def forward(self, images, texts, tokenize):
        clip_visual_encoder = self.clip_model.visual
        image_features = self.med_vis_extractor(images, clip_visual_encoder)
        texts = [tokenize(text[0]) for text in texts]
        texts = torch.cat(texts).to('cuda')
        text_features = self.clip_text_adapter(texts)
        return image_features, text_features
