import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from pyskl.datasets import build_dataset, build_dataloader

# Setting seed for reproducibility
SEED = 42

BATCH_SIZE = 32
INPUT_SHAPE = (56, 56, 56, 17)
NUM_CLASSES = 60

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 200

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 768
NUM_HEADS = 8
NUM_LAYERS = 8

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        results = self.model.evaluate(self.test_data, verbose=0)
        print(f"Epoch {epoch+1}, Test Accuracy: {results[1]*100:.2f}%, Test Top 5 Accuracy: {results[2]*100:.2f}%")

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

def count_model_params(model):
    return np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])

dataset = read_pkl('data/action/ntu60_hrnet.pkl')
dataset_type = 'PoseDataset'
ann_file = 'data/action/ntu60_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=56),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=56),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=False, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=1,
    test_dataloader=dict(videos_per_gpu=32),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', pipeline=train_pipeline)),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=test_pipeline))

print('Loading dataset...')

train_dataset = [build_dataset(data['train'])]
test_dataset = [build_dataset(data['test'], dict(test_mode=True))]

train_dataloader_setting = dict(
    videos_per_gpu=data.get('videos_per_gpu', 1),
    workers_per_gpu=data.get('workers_per_gpu', 1),
    persistent_workers=data.get('persistent_workers', False))
train_dataloader_setting = dict(train_dataloader_setting,
                            **data.get('train_dataloader', {}))
    
test_dataloader_setting = dict(
        videos_per_gpu=data.get('videos_per_gpu', 1),
        workers_per_gpu=data.get('workers_per_gpu', 1),
        persistent_workers=data.get('persistent_workers', False),
        shuffle=False)
test_dataloader_setting = dict(test_dataloader_setting,
                            **data.get('test_dataloader', {}))

train_data_loaders = [
    build_dataloader(ds, **train_dataloader_setting) for ds in train_dataset
]

test_data_loaders = [
    build_dataloader(ds, **test_dataloader_setting) for ds in test_dataset
]

def pytorch_to_tf_dataset(pytorch_dataloader):
    def generator():
        for batch_data in pytorch_dataloader:
            # Extracting images and labels
            imgs, labels = batch_data['imgs'], batch_data['label']

            # Convert PyTorch tensor to NumPy
            imgs_np = imgs.numpy()
            labels_np = labels.numpy()

            # Ensure labels are a 1D array
            labels_np = labels_np.squeeze()
            imgs_np = imgs_np.squeeze(1)

            # Permute the dimensions to match TensorFlow format
            imgs_np = imgs_np.transpose(0, 2, 3, 4, 1)  

            yield imgs_np, labels_np

    # Create TensorFlow dataset
    return tf.data.Dataset.from_generator(
        generator, 
        output_signature=(
            tf.TensorSpec(shape=(None, 56, 56, 56, 17), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )

# Convert PyTorch DataLoader to TensorFlow Dataset
trainloader_tf = pytorch_to_tf_dataset(train_data_loaders[0])
testloader_tf = pytorch_to_tf_dataset(test_data_loaders[0])

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )
    print(f"Total number of parameters in the model: {count_model_params(model)}")

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    # Initialize the custom callback
    test_callback = TestAccuracyCallback(testloader_tf)

    # Train the model.
    _ = model.fit(trainloader_tf, epochs=EPOCHS, callbacks=[test_callback])#, validation_data=validloader)

    _, accuracy, top_5_accuracy = model.evaluate(testloader_tf)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return model


model = run_experiment()