use crate::{Backend, Graph, Input, Internal, Op};
use rand_core::RngCore;
use std::cell::RefCell;
use std::ops::{Add, Sub};
use std::rc::Rc;

/// Stores the operations done to arrive at the final Tensor value in its [Graph]
/// The input field store where the input comes from
pub struct Tensor {
    graph: Rc<RefCell<Graph>>,
    input: Input,
}

impl Tensor {
    pub fn train_const(shape: Vec<usize>, value: f64) -> Self {
        let mut graph: Graph = Default::default();
        graph.ops.push(Op::TrainConst(shape, value));
        Tensor {
            graph: Rc::new(RefCell::new(graph)),
            input: Input::Internal(Internal { node: 0, output: 0 }),
        }
    }

    pub fn squared(&self) -> Self {
        let graph = self.graph.clone();
        let node = graph.borrow_mut().append(Op::Square(self.input.clone()));
        Self {
            graph,
            input: Input::Internal(Internal { node, output: 0 }),
        }
    }

    /// Creates the state for the tensor.
    pub fn gen_state<B>(&self, backend: &B, rng: impl RngCore) -> Result<B::State, B::Error>
    where
        B: Backend,
    {
        backend.state(&self.graph.borrow(), rng)
    }

    /// Evaluate the tensor.
    pub fn eval<B>(
        &self,
        backend: &B,
        state: &B::State,
        inputs: &B::TensorDict,
    ) -> Result<B::Tensor, B::Error>
    where
        B: Backend,
    {
        backend
            .forward(&self.graph.borrow(), state, inputs, self.input.clone())
            .map(|(output, _)| output)
    }

    /// Train the graph with this tensor as a loss function using gradient descent.
    ///
    /// Must be provided a way to convert the loss tensor into a `f32` and a `f32` to a tensor.
    ///
    /// Returns the loss before training.
    pub fn gradient_descent<B>(
        &self,
        backend: &B,
        state: &mut B::State,
        inputs: &B::TensorDict,
        learning_rate: f32,
        tensor_loss: fn(B::Tensor) -> f32,
        delta_tensor: fn(f32) -> B::Tensor,
    ) -> Result<f32, B::Error>
    where
        B: Backend,
    {
        // Perform the forward pass.
        let (output, internal) =
            backend.forward(&self.graph.borrow(), state, inputs, self.input.clone())?;

        // Extract the loss and compute the output delta.
        let loss = tensor_loss(output);
        let output_delta = delta_tensor(-learning_rate * loss);

        // Propogate the output delta back through the network.
        let delta = backend.backward(
            &self.graph.borrow(),
            state,
            &internal,
            inputs,
            self.input.clone(),
            output_delta,
        )?;

        // Train the network.
        backend.train(state, &delta)?;

        // Return the loss.
        Ok(loss)
    }
}

/// Creates a Tensor with an empty [Graph], no Ops. Its value will be fetched from the
/// [Backend::TensorDict] using the provided String as key
impl From<&str> for Tensor {
    fn from(s: &str) -> Tensor {
        Tensor {
            graph: Default::default(),
            input: s.into(),
        }
    }
}

/// Merges the graph associated with b tensor with the graph associated with a,
/// appending b into the end of a.
/// Shifts the b inputs by the length of a, so they "point" to the right place still
/// Puts it all into a new Tensor and returns it
fn merge2_1(a: Tensor, b: Tensor, make_op: impl Fn(Input, Input) -> Op) -> Tensor {
    let a_graph = a.graph;
    let a_input = a.input;
    let a_with_b_merged = a_graph
        .borrow_mut()
        .merge_input(b.graph.borrow().clone(), b.input);
    let node = a_graph
        .borrow_mut()
        .append(make_op(a_input, a_with_b_merged));
    Tensor {
        graph: a_graph,
        input: Input::Internal(Internal { node, output: 0 }),
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        merge2_1(self, rhs, Op::Add)
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        merge2_1(self, rhs, Op::Sub)
    }
}
