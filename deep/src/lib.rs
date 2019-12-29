#[macro_use]
extern crate strum_macros;

mod tensor;

pub use tensor::Tensor;

use rand_core::RngCore;

/// References a tensor which is produced as an output of an operation stored in the graph
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Internal {
    /// The result of which [Op] the tensor is.
    pub node: usize,
    /// The specific output to pull from, the [Op] could result in multiple outputs
    pub output: usize,
}

impl Internal {
    fn shift_inputs(&mut self, shift: usize) {
        self.node += shift;
    }
}

#[derive(Clone, Debug, EnumDiscriminants)]
#[strum_discriminants(name(OpTy), derive(Hash))]
pub enum Op {
    Add(Input, Input),
    Sub(Input, Input),
    Square(Input),
    TrainConst(Vec<usize>, f64),
}

impl Op {
    fn shift_inputs(&mut self, shift: usize) {
        match self {
            Self::Add(a, b) => {
                a.shift_inputs(shift);
                b.shift_inputs(shift);
            }
            Self::Sub(a, b) => {
                a.shift_inputs(shift);
                b.shift_inputs(shift);
            }
            Self::Square(a) => {
                a.shift_inputs(shift);
            }
            Self::TrainConst(..) => {}
        }
    }
}

/// Inputs which are used by an operation [Op].
///
/// They can either be a:
/// * String which will be used to fetch the actual Tensor from a dictionary later
/// (a HashMap<String, Tensor> for example)
/// * Internal which holds the index of the node in the [Graph] from where to get the input from
#[derive(Clone, Debug)]
pub enum Input {
    /// A String corresponding to the Key to use when fetching the actual Tensor from the feed dict.
    /// For example, if we had a HashMap<String, Tensor>
    Feed(String),
    /// An input from another node in the [Graph].
    Internal(Internal),
}

impl Input {
    fn shift_inputs(&mut self, shift: usize) {
        if let Self::Internal(n) = self {
            n.shift_inputs(shift);
        }
    }
}

impl From<&str> for Input {
    fn from(s: &str) -> Input {
        Input::Feed(s.to_owned())
    }
}

#[derive(Clone, Default, Debug)]
pub struct Graph {
    /// A series of [Op]s referring to each other's outputs for their input.
    pub ops: Vec<Op>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&mut self, other: Graph) {
        let current = self.ops.len();
        self.ops.extend(other.ops);
        for op in &mut self.ops[current..] {
            op.shift_inputs(current);
        }
    }

    pub fn merge_input(&mut self, other: Graph, mut input: Input) -> Input {
        let current = self.ops.len();
        self.merge(other);
        input.shift_inputs(current);
        input
    }

    /// Returns the node index of the appended [Op].
    pub fn append(&mut self, op: Op) -> usize {
        self.ops.push(op);
        self.ops.len() - 1
    }
}

pub trait Backend {
    /// The inputs where to get the actual Tensors from.
    /// Could be for example a HashMap<String, Tensor>
    type TensorDict;
    /// Same internal storage.
    /// Could be used, for example, to stores all the intermediary computations of the whole graph.
    type InternalStorage;
    /// The actual Tensor type used by this backend, for example `CudaFloat` or `ArcArray`
    type Tensor;
    /// The delta stores a map from nodes in the graph to their received gradient.
    type Delta;
    /// State contains all state data (internal tensors that are being trained or static).
    type State;
    type Error;

    /// Generates the initial state for a graph.
    fn state<R>(&self, graph: &Graph, rng: R) -> Result<Self::State, Self::Error>
    where
        R: RngCore;

    /// Gets the output of solving the requested tensor.
    fn forward(
        &self,
        graph: &Graph,
        state: &Self::State,
        inputs: &Self::TensorDict,
        tensor: Input,
    ) -> Result<(Self::Tensor, Self::InternalStorage), Self::Error>;

    /// Propagates a delta from the output back to the input via chain rule
    /// and produces a `Delta` that can be used to update the graph
    /// with an optimizer. The `Delta` contains all the dE/dx of all internal
    /// variables.
    fn backward(
        &self,
        graph: &Graph,
        state: &Self::State,
        internal: &Self::InternalStorage,
        inputs: &Self::TensorDict,
        tensor: Input,
        output_delta: Self::Tensor,
    ) -> Result<Self::Delta, Self::Error>;

    /// Applies a delta to the graph's state.
    fn train(&self, state: &mut Self::State, delta: &Self::Delta) -> Result<(), Self::Error>;
}
