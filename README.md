# Deep
Deep learning crate for Rust.

Most of the discussions happen at Discord: https://discord.gg/yqmFtwZ at the moment.

# Architecture Overview

Currently a *Tensor* is represented by a series of computations (*Ops* inside *Graph*) performed on an *Input*.

The *Input* can be the value inside a dictionnaire (TensorDict) or the result of another *Op*.

In the *Input*, *Node* represents which *Op* in the *Graph* to get the result from. Output represents which of the outputs from the computation to get.

<img alt=Architecture src="https://github.com/tiberiusferreira/deep/blob/initial_graph/docs/Architecture.png?raw=true" width="600"/>

## Example

<img alt=Architecture src="https://github.com/tiberiusferreira/deep/blob/initial_graph/docs/Example.png" width="600"/>
