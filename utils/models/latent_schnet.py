from schnetpack.atomistic import Atomwise
from schnetpack.nn.blocks import MLP
from schnetpack.nn import Dense
from schnetpack import Properties
import schnetpack
import torch


class LatentAtomwise(Atomwise):
    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
    ):

        if outnet is None:
            outnet = torch.nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                LatentMLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        elif isinstance(outnet[1], LatentMLP):
            pass
        elif isinstance(outnet[1], schnetpack.nn.blocks.MLP):
            outnet = torch.nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                LatentMLP.from_MLP(outnet[1]),
            )
        super(LatentAtomwise, self).__init__(
            n_in=n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            property=property,
            contributions=contributions,
            derivative=derivative,
            negative_dr=negative_dr,
            stress=stress,
            create_graph=create_graph,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
            outnet=outnet,
        )

    @classmethod
    def from_Atomwise(cls, atomwise):
        assert isinstance(atomwise, Atomwise)
        new = cls(
            atomwise.out_net[1].n_neurons[0],
            n_out=atomwise.out_net[1].n_neurons[-1],
            n_layers=atomwise.n_layers,
            property=atomwise.property,
            contributions=atomwise.contributions,
            derivative=atomwise.derivative,
            negative_dr=atomwise.negative_dr,
            stress=atomwise.stress,
            create_graph=atomwise.create_graph,
            mean=atomwise.standardize.mean,
            stddev=atomwise.standardize.stddev,
            atomref=None,
            outnet=atomwise.out_net,
        )
        new.atomref = atomwise.atomref
        _ = new.load_state_dict(atomwise.state_dict())
        return new

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        yi = self.out_net(inputs)
        yi[0] = self.standardize(yi[0])

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi[0] = yi[0] + y0

        yi[0] = self.atom_pool(yi[0], atom_mask)
        return yi


class LatentMLP(MLP):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=None,
        n_layers=2,
        activation=schnetpack.nn.activations.shifted_softplus,
    ):
        super(LatentMLP, self).__init__(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

        layers = [
            LatentDense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        layers.append(
            LatentDense(self.n_neurons[-2], self.n_neurons[-1], activation=None)
        )
        self.out_net = torch.nn.Sequential(*layers)

    @classmethod
    def from_MLP(cls, mlp):
        assert isinstance(mlp, MLP)
        new = cls(
            mlp.n_neurons[0],
            mlp.n_neurons[-1],
            n_hidden=None,
            n_layers=len(mlp.n_neurons) - 1,
        )
        _ = new.load_state_dict(mlp.state_dict())
        return new


class LatentDense(Dense):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=schnetpack.nn.activations.shifted_softplus,
    ):
        super(LatentDense, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation=activation,
        )

    @classmethod
    def from_Dense(cls, den):
        assert isinstance(den, Dense)
        new = cls(
            den.in_features,
            den.out_features,
            bias=den.bias is not None,
            activation=den.activation,
        )
        _ = new.load_state_dict(den.state_dict())
        return new

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        base_y = super(Dense, self).forward(inputs[0])
        if self.activation:
            y = self.activation(base_y)
        else:
            y = base_y
        return_list = [y, base_y]
        for i in range(len(inputs) - 1):
            return_list.append(inputs[i + 1])
        return return_list
