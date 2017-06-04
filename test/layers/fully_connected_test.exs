defmodule Neuro.Layers.FullyConnectedTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Layers.FullyConnected
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Cuda.Graph

    def __graph__(graph) do
      graph |> chain(:fully, FullyConnected, graph.assigns.options) |> close()
    end

    defdelegate __assigns__(id, opts, env), to: FullyConnected
    defdelegate __pins__(assings), to: FullyConnected
  end

  describe "fully connected layer" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Wrapper
    @tag options: [size: 8, out_size: 2]
    @tag shared: %{
      weights: %{fully: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]},
      biases:  %{fully: [1.0, 0.0]}
    }
    test "simple fully connected", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [21.4, 12.5]
    end
  end
end
