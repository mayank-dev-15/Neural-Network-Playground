import torch
import torch.nn as nn


class ArchitectureRegistry:
    _architectures = {}

    @classmethod
    def register(cls, name, builder_fn=None):
        def decorator(fn):
            cls._architectures[name] = fn
            return fn
        if builder_fn is not None:
            cls._architectures[name] = builder_fn
            return builder_fn
        return decorator

    @classmethod
    def get(cls, name):
        return cls._architectures.get(name)

    @classmethod
    def list(cls):
        return list(cls._architectures.keys())

    @classmethod
    def build(cls, name, **kwargs):
        builder = cls.get(name)
        if builder is None:
            raise ValueError(f"Unknown architecture: {name}. Available: {cls.list()}")
        return builder(**kwargs)


class MLPBuilder:
    @staticmethod
    def build(input_dim, hidden_dims, output_dim, activation='relu', dropout=0.0):
        layers = []
        prev = input_dim
        activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
        }.get(activation, nn.ReLU)

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def build_classifier(input_dim, hidden_dims, num_classes, **kwargs):
        model = MLPBuilder.build(input_dim, hidden_dims, num_classes, **kwargs)
        return model

    @staticmethod
    def build_autoencoder(input_dim, hidden_dims, encoding_dim):
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.ReLU())
            prev = h
        encoder_layers.append(nn.Linear(prev, encoding_dim))
        encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Linear(encoding_dim, hidden_dims[-1]), nn.ReLU()]
        prev = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.ReLU())
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        decoder = nn.Sequential(*decoder_layers)

        return nn.Sequential(encoder, decoder)


class CNNBuilder:
    @staticmethod
    def build_2d(in_channels, conv_configs, fc_dims, num_classes, input_size=32):
        layers = []
        cin = in_channels
        current_size = input_size
        for cc in conv_configs:
            cout = cc.get('out_channels', 16)
            kernel = cc.get('kernel', 3)
            stride = cc.get('stride', 1)
            padding = cc.get('padding', 0)
            pool = cc.get('pool', None)
            layers.append(nn.Conv2d(cin, cout, kernel, stride, padding))
            layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(pool))
                current_size = (current_size - pool) // pool + 1
            else:
                current_size = (current_size + 2 * padding - kernel) // stride + 1
            cin = cout

        flattened = cin * current_size * current_size
        classifier = []
        prev = flattened
        for fd in fc_dims:
            classifier.append(nn.Linear(prev, fd))
            classifier.append(nn.ReLU())
            prev = fd
        classifier.append(nn.Linear(prev, num_classes))

        return nn.Sequential(
            nn.Sequential(*layers),
            nn.Flatten(),
            nn.Sequential(*classifier),
        )

    @staticmethod
    def build_simple(in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )


class RNNBuilder:
    @staticmethod
    def build(input_size, hidden_size, num_layers, output_size, cell='lstm', bidirectional=False, dropout=0.0):
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU}.get(cell, nn.LSTM)
        num_directions = 2 if bidirectional else 1
        rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True,
                      bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        fc = nn.Linear(hidden_size * num_directions, output_size)

        class RNNWrapper(nn.Module):
            def __init__(self, rnn, fc):
                super().__init__()
                self.rnn = rnn
                self.fc = fc

            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.fc(out[:, -1, :])
                return out

        return RNNWrapper(rnn, fc)


@ArchitectureRegistry.register('mlp')
def build_mlp(**kwargs):
    return MLPBuilder.build(**kwargs)

@ArchitectureRegistry.register('cnn')
def build_cnn(**kwargs):
    return CNNBuilder.build_simple(**kwargs)

@ArchitectureRegistry.register('rnn')
def build_rnn(**kwargs):
    return RNNBuilder.build(**kwargs)

@ArchitectureRegistry.register('autoencoder')
def build_autoencoder(**kwargs):
    return MLPBuilder.build_autoencoder(**kwargs)
