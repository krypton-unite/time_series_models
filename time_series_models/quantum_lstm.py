import torch
import torch.nn as nn

import pennylane as qml

class QuantumLSTM(nn.Module):
    def __init__(self, 
                input_dim=1,
                hidden_dim=4,
                output_dim=1, 
                num_layers=1,
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = input_dim
        self.output_dim = output_dim
        self.concat_size = self.n_inputs + self.output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.dev = qml.device("default.qubit", wires=self.hidden_dim)
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.hidden_dim))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.hidden_dim))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.hidden_dim)]
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")

        weight_shapes = {"weights": (num_layers, hidden_dim)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({num_layers}, {hidden_dim})")

        self.clayer_in = torch.nn.Linear(self.concat_size, hidden_dim)
        self.VQC = [qml.qnn.TorchLayer(self.qlayer, weight_shapes) for _ in range(4)]
        self.clayer_out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.init_states = None
        #self.clayer_out = [torch.nn.Linear(hidden_dim, self.output_dim) for _ in range(4)]

    def forward(self, x):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        if self.init_states is None:
            h_t = torch.zeros(batch_size, self.output_dim).to(x.device)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.output_dim).to(x.device)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = self.init_states
            # h_t = h_t[0]
            # c_t = c_t[0]
            h_t = h_t.detach()
            c_t = c_t.detach()

        hidden_seq = []
        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC[0](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC[1](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC[2](y_t)))     # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC[3](y_t)))  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        self.init_states = (h_t, c_t)
        return hidden_seq #, (h_t, c_t)