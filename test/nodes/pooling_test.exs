defmodule Neuro.Nodes.PoolingTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Pooling
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Neuro.Nodes.Base, proto: Cuda.Graph

    def __graph__(graph) do
      graph |> chain(:network, Pooling, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: Pooling
  end


  describe "pooling node" do
    setup do
      log_level = Logger.level()
      Logger.configure(level: :error)
      on_exit(fn -> Logger.configure(level: log_level) end)
    end

    test "pooling: 2x2, stride: 2x2" do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      opts = [network: Wrapper,
              network_options: [size: {4, 4}, pooling: {2, 2}, stride: {2, 2}]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        0.6, 0.8,
        1.0, 0.7
      ]
    end

    test "pooling: 2x2, stride: 1x1" do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      opts = [network: Wrapper,
              network_options: [size: {4, 4}, pooling: 2, stride: 1]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        0.6, 0.7, 0.8,
        1.0, 0.7, 0.8,
        1.0, 0.6, 0.7
      ]
    end
  end
end
