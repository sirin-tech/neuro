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

  # TODO: BackNetwork needed only to run tests in parallel because network
  #       module name used to name Registry. To resolve this we should add
  #       something like `name` option that will override default naming
  defmodule BackNetwork do
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
    setup ~w(disable_logging)a

    test "simple network" do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.7,
           1.0, 0.1, 0.2]

      shared = %{
        weights: %{
          conv: [1.0, 2.0, 3.0, 4.0],
          fc: [0.1, 0.2, 0.3, 0.4,
               1.0, 2.0, 3.0, 4.0,
               10.0, 20.0, 30.0, 40.0]
        },
        biases: %{
          conv: [0.0],
          fc: [0.0, 0.0, 0.0]
        }
      }

      {:ok, _} = Network.start_link(SimpleNetwork, shared: %{shared: shared})
      {:ok, o} = Network.run(SimpleNetwork, %{input: i})
      # conv output: [[4.4, 5.4, 5.1, 3.1]]
      assert round!(o.output) == [4.3, 42.9, 429.0]
    end

    test "back propagation" do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.7,
           1.0, 0.1, 0.2]
      r = [0.3, 40.9, 425.0]

      shared = %{
        weights: %{
          conv: [1.0, 2.0, 3.0, 4.0],
          fc: [0.1, 0.2, 0.3, 0.4,
               1.0, 2.0, 3.0, 4.0,
               10.0, 20.0, 30.0, 40.0]
        },
        biases: %{
          conv: [0.0],
          fc: [0.0, 0.0, 0.0]
        }
      }

      {:ok, _} = Network.start_link(BackNetwork, shared: %{shared: shared}, network_options: [type: :training])
      {:ok, o} = Network.run(BackNetwork, %{input: i, reply: r})
      assert round!(o.error) == 18.0
      #IO.inspect(o)
    end
  end
end
