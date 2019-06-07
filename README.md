# PyNN Object Serialisation

This repository shows one approach to saving a PyNN simulation using a pair of JSON and NPZ files representing the network architecture and connectivity respectively. The current implementation specifically targets the [sPyNNaker](https://github.com/spinnakermanchester/spynnaker) backend for PyNN

### Supported Objects

1. LIF neurons
2. Static Synapses
3. Array-based and Poisson spike sources

### Use cases

1. Once a PyNN network is automatically converted from an ANN to an SNN, this tool can ensure the correct saving and re-loading of the network.
