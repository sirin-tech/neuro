defmodule Neuro.NetworkTest do
  use ExUnit.Case
  alias Neuro.Network
  alias Neuro.Layers

  defmodule SimpleNetwork do
    use Network

    def graph(graph) do
      graph
      |> chain(:conv, Layers.Convolution)
      |> chain(:fc,   Layers.FullyConnected)
      |> close()
    end
  end

  describe "Network" do
    test "simple network" do
    end
  end
end
