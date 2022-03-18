# Process
The entire process is split into data processing, model definition and training bbox, training segmentation
### Data preparation
The dataset is downloaded and we see that the folder structure is
 -> Annotations
 -> images
 -> masks
We write a script to create the bounding box from the mask. For this we take pytorch's standard module **from torchvision.ops import masks_to_boxes** and build on it. We also use assertion statements to check if shape of images and matching along with filename to avoid using auto image rotation using metadata opencv vs PIL issues. There are 
potentially 3 types of classes in the segmentation
* spallmask
* rebarmask
* crackmask

These are loaded seperately as different masks then merged. The bouding box is obtained for each of these classes.
While converting the entire thing to coco format is one approach another one is where the dataloader matches the output
We see for detr bbox following are the input types
```
{'area': tensor([[141581.2188]]),
 'boxes': tensor([[[0.3353, 0.5104, 0.6706, 0.4080]]]),
 'image_id': tensor([[3174]]),
 'iscrowd': tensor([[False]]),
 'labels': tensor([[1]]),
 'orig_size': tensor([[570, 756]]),
 'size': tensor([[704, 735]])}
```
And output from ourdata loader is 
```
{'boxes': tensor([[[  0.,   0., 185., 254.],
          [  0.,   0., 185., 254.],
          [  0.,   0., 185., 254.]]]),
 'labels': tensor([[1, 1, 1]])}
```

### Model
A pretrained detr model will be used and transfer learning with low lr will be utilized to train the model on custom dataset. For transfer learning two parameters are deleted in the checkpoint weights.
```
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
del checkpoint["model"]["query_embed.weight"]
```
The number of class is chaged and the checkpoint weights are loaded. The standard script is used of training.

# TODO
Resize operation needs to be fixed with albumentations.

## Questions
**We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**

This encoded image is taken from the output of the res-net Backbone after passing through a 1x1 convolutional network. The input image in this case is 3xWxH. The backbone activation map output is CxH/32x/32 this is passed through a 1x1 network and we get **dxH/32xW/32.** The section of the code that does this is

```python
class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads = 8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.backbone = resnet50()
        del self.backbone.fc
        # create coversion to dimension 
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
		def forward(self, inputs):
		        # propagate inputs through ResNet-50 up to avg-pool layer
		        x = self.backbone.conv1(inputs)
		        x = self.backbone.bn1(x)
		        x = self.backbone.relu(x)
		        x = self.backbone.maxpool(x)
		
		        x = self.backbone.layer1(x)
		        x = self.backbone.layer2(x)
		        x = self.backbone.layer3(x)
		        x = self.backbone.layer4(x)
		
		        # convert from 2048 to 256 feature planes for the transformer
		        h = self.conv(x)
```

For segmentation this output is sent to the encoder which retains the same dimension. This encoded image is sent to the segmentation Multi-Headed Attention block.

**We also send dxN Box embeddings to the Multi-Head Attention. We do something here to generate NxMxH/32xW/32 maps. (WHAT DO WE DO HERE?)**

Since the decoder is permutation invariant the input queries need to be different these N queries are called object queries. Where d is the hidden dimension of the transformer. The output of the decoder has dxN dimension which is fed into the segmentation Multi-Head Attention network along with the encoded image. The number of heads in the Multi-Headed attention network is M. 

To generate the NxMxH/32,W/32 the Multiheaded Attention block takes the Bbox, the output of the encoder layer and the output of the feature map to create a mask.

**Then we concatenate these maps with Res5 Block (WHERE IS THIS COMING FROM?)**

The attention maps are concatenated with the Res5 Block

```jsx
class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
		def forward(self, samples: NestedTensor):
				bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
```

```jsx
class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
```

**Then we perform the above steps (EXPLAIN THESE STEPS) And then we are finally left with the panoptic segmentation**

We use an FPN style CNN. We interpolate the inputs to the layer while passing through the FPN adapter layers and a pixel wise argmax (sigmoid with a threshold) is used to obtain the final output.

The final panoptic segmentation postprocessing code identifies each thing and stuff class and their ids and merges them into one single image.

