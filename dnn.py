import torch.nn as nn
import torch.optim as optim
import torch

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Use variable input_size
        self.fc2 = nn.Linear(hidden_size, output_size)  # Use variable output_size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LargerNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LargerNN, self).__init__()
        layers = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        # Add the final output layer
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class WeightedMSELoss(nn.Module):
    def __init__(self, target_weights):
        super(WeightedMSELoss, self).__init__()
        self.target_weights = target_weights

    def forward(self, output, target):
        mse_loss = torch.nn.functional.mse_loss(output, target, reduction='none')
        weighted_loss = mse_loss * self.target_weights
        return weighted_loss.mean()


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)  # Hidden size should match
        self.Ua = nn.Linear(hidden_size, hidden_size)  # Hidden size should match
        self.Va = nn.Linear(hidden_size, 1)  # Output a single score

    def forward(self, query, keys, values):
        # Apply linear transformations
        query_transformed = self.Wa(query)  # (batch_size, seq_length, hidden_size)
        keys_transformed = self.Ua(keys)  # (batch_size, seq_length, hidden_size)
        scores = self.Va(torch.tanh(query_transformed + keys_transformed))  # (batch_size, seq_length, 1)

        # Calculate attention weights
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_length, 1)

        # Compute context vector
        context = torch.bmm(attention_weights, values)  # (batch_size, 1, hidden_size)
        return context, attention_weights


class LargerNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, attention_hidden_size):
        super(LargerNNWithAttention, self).__init__()
        self.attention = Attention(attention_hidden_size)
        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))  # Ensure output_size matches the target size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing

        batch_size, seq_length, _ = x.size()

        query = x
        keys = x
        values = x

        context, attention_weights = self.attention(query, keys, values)
        context = context.squeeze(1)  # Remove the sequence dimension

        output = self.network(context)
        return output, attention_weights

