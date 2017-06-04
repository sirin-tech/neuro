defmodule Neuro.Nodes.PoolingTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Pooling
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph |> chain(:pooling, Pooling, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: Pooling
  end


  describe "pooling node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Wrapper
    @tag options: [size: {4, 4}, pooling: {2, 2}, stride: {2, 2}]
    test "pooling: 2x2, stride: 2x2", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        [0.6, 0.8],
        [1.0, 0.7]
      ]
    end

    @tag graph: Wrapper
    @tag options: [size: {4, 4}, pooling: 2, stride: 1]
    test "pooling: 2x2, stride: 1x1", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        [0.6, 0.7, 0.8],
        [1.0, 0.7, 0.8],
        [1.0, 0.6, 0.7]
      ]
    end
  end
end
