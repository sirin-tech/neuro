defmodule Neuro.NetworkTest do
  use ExUnit.Case
  alias Neuro.Network
  alias Neuro.Layers
  import Neuro.Test.NodesHelpers

  defmodule SimpleNetwork do
    use Network

    input  {3, 3}
    output 3

    def graph(graph) do
      graph
      |> chain(:conv, Layers.Convolution, kernel_size: {2, 2})
      |> chain(:fc,   Layers.FullyConnected, out_size: 3)
      |> close()
    end
  end

  describe "Network" do
    setup do
      log_level = Logger.level()
      Logger.configure(level: :error)

      weights = %{
        conv: [1.0, 2.0, 3.0, 4.0],
        fc: [0.1, 0.2, 0.3, 0.4,
             1.0, 2.0, 3.0, 4.0,
             10.0, 20.0, 30.0, 40.0]
      }
      biases = %{
        conv: [0.0, 0.0, 0.0, 0.0],
        fc: [0.0, 0.0, 0.0]
      }

      network = SimpleNetwork.start_link(weights: weights, biases: biases)

      on_exit(fn ->
        Logger.configure(level: log_level)
      end)
      [network: network]
    end

    test "simple network", _ctx do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.7,
           1.0, 0.1, 0.2]
      {:ok, o} = SimpleNetwork.run(%{input: i})
      # conv output: [[4.4, 5.4, 5.1, 3.1]]
      assert round!(o.output) == [4.3, 42.9, 429.0]
    end
  end
end
