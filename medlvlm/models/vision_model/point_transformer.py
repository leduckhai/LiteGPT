import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, GELU, Dropout
import numpy as np


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def fps(xyz, npoint, mask = None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    if mask is not None:
        distance = distance.masked_fill(mask == 0, 0)

    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        if mask is not None:
            dist = dist.masked_fill(mask == 0, 1e10)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz, mask = None):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    if mask is not None:
        sqrdists = sqrdists.masked_fill(mask.unsqueeze(1) == 0, float('inf'))
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        batch_size, num_points, _ = xyz.shape

        if C > 3:
            data = xyz
            xyz = data[:,:,:3]
            rgb = data[:, :, 3:]
        
        
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
          # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(B * N, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center


def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals


class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    
# MLP module used within Transformer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

# Transformer Block containing Attention and MLP
class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super(Block, self).__init__()
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=GELU, drop=drop)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.drop_path = Dropout(drop_path)

    def forward(self, x):
        # print("x in block:", x.shape)
        x_norm1 = self.norm1(x)
        # print("x_norm1 in block:", x_norm1.shape)
        x = x + self.drop_path(self.attn(x_norm1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class PointTransformer(nn.Module):
    def __init__(self,
                    transdim = 384,
                    depth = 12,
                    drop_path_rate = 0.1,
                    cls_dim = 40,
                    num_heads = 6,
                    group_size = 32,
                    num_group = 512,
                    encoder_dims = 256,
                    point_dims = 6,
                    projection_hidden_layer = 2,
                    projection_hidden_dim = [1024, 2048],
                    input_points = 200000,
                    num_classes = 41,
                    use_max_pool=False):
        super(PointTransformer, self).__init__()


        self.trans_dim = transdim
        self.num_features = self.trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.use_max_pool = use_max_pool

        self.group_size = group_size
        self.num_group = num_group
        self.point_dims = point_dims
        self.num_classes = num_classes
        self.input_points = input_points

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        # Position embedding for center
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        try:
            self.blocks = torch.compile(TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.num_heads
            ))
        except:
            print("Torch Compile is NOT Supported")
            self.blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.num_heads
            )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.fc_seg = nn.ConvTranspose1d(self.num_group + 1, self.input_points, kernel_size=1, stride=1)
        self.seg_to_classes = nn.Linear(self.trans_dim, self.num_classes)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return x
        
        # final input
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f
    

def create_point_transformer(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = PointTransformer()
    if precision == "fp16":
        model = model.half()
    return model