import tensorflow as tf 


class Encoder(tf.keras.Model):
    def __init__(self,
                 num_heads, 
                 dim, 
                 hidden_dim=2048, 
                 dropout=0.1, 
                 activation="relu", 
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
   
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim,
            dropout=dropout,
            name="self_attn")
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout, name="dropout1")
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, name="norm1")
    
        self.linear1 = tf.keras.layers.Dense(units=hidden_dim, activation=activation, name="linear1")
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout, name="/dropout2")
        self.linear2 = tf.keras.layers.Dense(units=hidden_dim, name="/linear2")
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout, name="/dropout3")
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, name="/norm2")
    
    def call(self, src, src_mask=None, pos_embed=None, training=None):
        query = key = src if pos_embed is None else pos_embed + src
        src2 = self.self_attn(query=query, key=key, value=src, mask=src_mask, training=training)
        src += self.dropout1(src2, training=training)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.linear1(src), training=training))
        src += self.dropout3(src2, training=training)
        src = self.norm2(src)

        return src 


class Decoder(tf.keras.Model):
    def __init__(self, 
                 dim, 
                 num_heads,
                 hidden_dim=2048, 
                 dropout=0.1, 
                 activation="relu",
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim,
            dropout=dropout,
            name="/self_attn")
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout, name="dropout1")(x)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, name="norm1")(x)
        
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim,
            dropout=dropout,
            name="multihead_attn")
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout, name="dropout2")
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, name="norm2")

        self.linear1 = tf.keras.layers.Dense(units=hidden_dim, activation=activation, name="linear1")
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout, name="dropout3")
        self.linear2 = tf.keras.layers.Dense(units=hidden_dim, name="linear2")
        self.dropout4 = tf.keras.layers.Dropout(rate=dropout, name="dropout4")
        self.norm3 = tf.keras.layers.LayerNormalization(axis=-1, name="norm3")
    
    def call(self, target, memory, target_mask=None, memeory_mask=None, pos_embed=None, query_pos_embed=None, training=None):
        q = k = target if query_pos_embed is None else target + query_pos_embed

        target2 = self.self_attn(query=q, key=k, value=target, mask=target_mask, training=training)
        target += self.dropout1(target2, training=training)
        target = self.norm1(target)

        target2 = self.multihead_attn(
            query=target if query_pos_embed is None else target + query_pos_embed,
            key=memory if pos_embed is None else memory + pos_embed,
            value=memory,
            mask=memory,
            training=training)
        target += self.dropout2(target2)
        target = self.norm2(target)
        target2 = self.linear2(self.dropout2(self.linear1(target), training=training))
        target += self.dropout3(target2, training=training)
        target = self.norm2(target)

        return target



    


