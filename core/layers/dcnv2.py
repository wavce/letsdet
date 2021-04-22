import numpy as np 
import tensorflow as tf
# import torch 
# import torch.nn as nn


class DCNv2(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="same",
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 trainable=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DCNv2, self).__init__(**kwargs)

        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)

        if padding == "same":
            p = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 
            self.pad = tf.keras.layers.ZeroPadding2D((p, p), name="pad")
        self.conv = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=kernel_size,
                                           strides=kernel_size,
                                           use_bias=use_bias,
                                           trainable=trainable,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activation=activation,
                                           data_format=data_format,
                                           name="conv2d")
      
        self._nk = self.kernel_size[0] * self.kernel_size[1]

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)
    
    def get_config(self):
        config = (list(self.conv.get_config().items()) + 
                  list(super(DCNv2, self).get_config().items()))
        
        return dict(config)
    
    def build(self, input_shape):
        pixel_indices = self._get_pixel_indices(input_shape[1:3])
        kernel_indices = self._get_kernel_position()
        kernel_indices = tf.tile(kernel_indices, [input_shape[0], input_shape[1], input_shape[2], 1, 1])

        self.pixel_indices = tf.cast(pixel_indices, self.dtype)
        self.kernel_indices = tf.cast(kernel_indices, self.dtype)

        super(DCNv2, self).build(input_shape)
    
    def _get_pixel_indices(self, input_shape):
        with tf.name_scope("get_pixel_indices"):
            x, y = tf.meshgrid(tf.range(1, input_shape[1] * self.strides[1] + 1, self.strides[1]), 
                               tf.range(1, input_shape[0] * self.strides[0] + 1, self.strides[0]))
            indices = tf.reshape(
                tf.stack([y, x], -1), 
                [1, input_shape[0] // self.strides[0], input_shape[1] // self.strides[1], 1, 2])
            indices = tf.tile(indices, [1, 1, 1, self._nk, 1])

            return indices
    
    def _get_kernel_position(self):
        with tf.name_scope("get_kernel_position"):
            kx, ky = tf.meshgrid(tf.range(-((self.kernel_size[1] -1) // 2), self.kernel_size[1] // 2 + 1), 
                                 tf.range(-((self.kernel_size[0] -1) // 2), self.kernel_size[0] // 2 + 1)) 

            k_indices = tf.reshape(tf.stack([tf.reshape(ky, [-1]), tf.reshape(kx, [-1])], -1), [1, 1, 1, self._nk, 2])
            
            return k_indices
    
    def _get_x(self, x, inds_y, inds_x, iw, chn, oh, ow, batch_dim_indices):
        inds = inds_y * tf.cast(iw, inds_y.dtype) + inds_x
        inds += batch_dim_indices
        inds = tf.reshape(inds, [-1])
        x = tf.reshape(x, [-1, chn])
        feats = tf.gather(x, tf.cast(inds, tf.int32))
        feats = tf.reshape(feats, [-1, oh, ow, self._nk, chn])

        return feats

    def call(self, inputs, offset, modulation=None):
        offset_shape = tf.shape(offset)
        b = offset_shape[0]
        oh = offset_shape[1]
        ow = offset_shape[2]
               
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        
        ih = tf.shape(x)[1]
        iw = tf.shape(x)[2]
        chn = tf.shape(x)[3]
        
        offset = tf.stack([offset[..., :self._nk], offset[..., self._nk:]], -1)
        p = offset + self.pixel_indices + self.kernel_indices
        y0x0 = tf.math.floor(p)
        y1x1 = y0x0 + 1.
        
        y0 = tf.clip_by_value(y0x0[..., 0], 0, tf.cast(ih - 1, y0x0.dtype))
        x0 = tf.clip_by_value(y0x0[..., 1], 0, tf.cast(iw - 1, y0x0.dtype))
        y1 = tf.clip_by_value(y1x1[..., 0], 0, tf.cast(ih - 1, y1x1.dtype))
        x1 = tf.clip_by_value(y1x1[..., 1], 0, tf.cast(iw - 1, y1x1.dtype))
        # y0x1 = tf.stack([y0x0[..., 0], y1x1[..., 1]], -1)
        # y1x0 = tf.stack([y1x1[..., 0], y0x0[..., 1]], -1)
        
        batch_dim_offset = tf.cast(ih * iw, y0x0.dtype)
        batch_dim_indices = (
            tf.reshape(tf.range(b, dtype=y0x0.dtype) * batch_dim_offset, [b, 1, 1, 1]) * 
            tf.ones([1, oh, ow, self._nk], dtype=y0x0.dtype))
        f00 = self._get_x(x, y0, x0, iw, chn, oh, ow, batch_dim_indices)  # left top
        f01 = self._get_x(x, y1, x0, iw, chn, oh, ow, batch_dim_indices)  # left bottom
        f10 = self._get_x(x, y0, x1, iw, chn, oh, ow, batch_dim_indices)  # right top
        f11 = self._get_x(x, y1, x1, iw, chn, oh, ow, batch_dim_indices)  # right bottom

        # neighboring feature points f0, f1, f2, and f3.
        # f(y, x) = [1 - ly, ly] * [[f00, f01], * [1 - lx, lx]^T
        #                           [f10, f11]]
        # f(y, x) = ((1 - ly) * (1 - lx)) f00 + ((1 - ly) * lx) f01 + (ly * (1 - lx))f10 + (lx * ly)f11
        # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
        py = tf.clip_by_value(p[..., 0], 0, tf.cast(ih - 1, p.dtype)) 
        px = tf.clip_by_value(p[..., 1], 0, tf.cast(iw - 1, p.dtype))
        py = tf.expand_dims(py, -1)
        px = tf.expand_dims(px, -1)
        y0 = tf.expand_dims(y0, -1)
        x0 = tf.expand_dims(x0, -1)
        ly = py - y0
        lx = px - x0
        w00 = (1. - ly) * (1. - lx)
        w01 = (1. - ly) * lx
        # w01 = ly * (1. - lx)
        # w10 = (1. - ly) * lx
        w10 = ly * (1. - lx)
        w11 = ly * lx
        features = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11

        if modulation is not None:
            features *= tf.expand_dims(modulation, -1)
        
        ks = self.kernel_size[0]
        features = tf.concat([features[..., s:s + ks, :] for s in range(0, self._nk, ks)], axis=1)
        features = tf.reshape(features, [b, oh * ks, ow * ks, chn])

        outputs = self.conv(features)

        return outputs
        

# class DeformConv2d(nn.Module):
#     def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
#         """
#         Args:
#             modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
#         """
#         super(DeformConv2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.zero_padding = nn.ZeroPad2d(padding)
#         self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

#         self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#         nn.init.constant_(self.p_conv.weight, 0)
#         self.p_conv.register_backward_hook(self._set_lr)

#         self.modulation = modulation
#         if modulation:
#             self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
#             nn.init.constant_(self.m_conv.weight, 0)
#             self.m_conv.register_backward_hook(self._set_lr)

#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

#     def forward(self, x):
#         offset = self.p_conv(x)
#         if self.modulation:
#             m = torch.sigmoid(self.m_conv(x))

#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2

#         if self.padding:
#             x = self.zero_padding(x)

#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)

#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1

#         q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

#         # clip p
#         p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)

#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt
        
#         # modulation
#         if self.modulation:
#             m = m.contiguous().permute(0, 2, 3, 1)
#             m = m.unsqueeze(dim=1)
#             m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
#             x_offset *= m

#         x_offset = self._reshape_x_offset(x_offset, ks)
#         out = self.conv(x_offset)

#         return out

#     def _get_p_n(self, N, dtype):
#         p_n_x, p_n_y = torch.meshgrid(
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

#         return p_n

#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = torch.meshgrid(
#             torch.arange(1, h*self.stride+1, self.stride),
#             torch.arange(1, w*self.stride+1, self.stride))
#         p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

#         return p_0

#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p

#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)

#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

#         return x_offset

#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

#         return x_offset


if __name__ == "__main__":
    import torch
    import numpy as np
    from detectron2.layers import ModulatedDeformConv

    np.random.seed(1)
    inp = np.random.uniform(0, 1, [1, 32, 32, 3])
    weights = np.random.uniform(0, 1, [3, 3, 3, 32])
    offset = np.random.uniform(0, 1, [1, 30, 30, 18])
    mask = np.random.uniform(0, 1, [1, 30, 30, 9])

    torch_dcn = ModulatedDeformConv(3, 32, 3, 1, 0, bias=False).cuda()

    torch_dcn.weight.data = torch.from_numpy(weights.transpose(3, 2, 0, 1)).contiguous().cuda()
    torch_inp = torch.from_numpy(inp.transpose(0, 3, 1, 2)).contiguous().cuda()
    torch_offset = torch.from_numpy(offset.transpose(0, 3, 1, 2)).contiguous().cuda()
    torch_mask = torch.from_numpy(mask.transpose(0, 3, 1, 2)).contiguous().cuda()
    x2 = torch_dcn(torch_inp, torch_offset, torch_mask).contiguous().permute(0, 2, 3, 1)

    tf_inp = tf.convert_to_tensor(inp, tf.float32)
    tf_w = tf.convert_to_tensor(weights, tf.float32)
    offset = tf.convert_to_tensor(offset, tf.float32)
    mask = tf.convert_to_tensor(mask, tf.float32)
    tf_dcn = DCNv2(32, 3, padding="valid", use_bias=False, kernel_initializer=tf.keras.initializers.Constant(tf_w))
    x1 = tf_dcn(tf_inp, offset, mask)
    print(x1.shape, x2.shape)

    print(x1[0, 0, 0])

    print(x2[0, 0, 0])