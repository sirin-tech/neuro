defmodule Neuro.Nodes.FullyConnectedTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.FullyConnected
  alias Cuda.Shared
  import Neuro.Test.NodesHelpers

  defmodule Inference do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph |> chain(:fc, FullyConnected, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: FullyConnected
  end

  defmodule BackPropagation do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph
      |> add(:fc, FullyConnected, graph.assigns.options)
      |> link(:output, {:fc, :output})
      |> link(:inference, {:fc, :inference})
      |> close()
    end

    defdelegate vars(opts, env), to: FullyConnected
  end

  describe "fully connected node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                            1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]]},
      biases:  %{network: [1.0, 0.0]}
    }
    test "simple fully connected", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [21.4, 12.5]
    end

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2]
    @tag shared: %{
      weights: %{network: [[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
                             1.5,  0.7,  6.4,  3.2,  2.1,  1.1,  0.1,  9.0]]},
      biases:  %{network: [0.0, 0.0]}
    }
    test "relu activation", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [0.0, 12.5]
    end

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2, training: true]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                            1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]]},
      biases:  %{network: [1.0, 0.0]},
      states:  %{network: [0.0, 0.0]}
    }
    test "saves neuron states in training mode", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, _o} = Cuda.Worker.run(worker, %{input: i})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])

      assert shared.states.network |> round!() == [21.4, 12.5]
    end

    @tag graph: BackPropagation
    @tag options: [size: 4, out_size: 3, back_propagation: true]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8]]},
      biases:  %{network: [0.0, 0.0, 0.0]},
      states:  %{network: [30.0, 0.0, 40.0]},
      speed:   0.05
    }
    test "back propagation", ctx do
      loss = [10.0, 15.0, 20.0]
      inf  = [11.0, 12.0, 13.0, 14.0]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{output: loss, inference: inf})
      assert o.input |> round! == [20.0, 32.0, 44.0, 56.0]
      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      assert shared.weights.network |> round!(2) == [-4.5,  -4.0,  -3.5,  -3.0,
                                                      0.1,   0.2,   0.3,   0.4,
                                                    -10.5, -11.4, -12.3, -13.2]
    end
  end
end
