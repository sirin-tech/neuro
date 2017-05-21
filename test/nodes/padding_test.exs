defmodule Neuro.Nodes.PaddingTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Padding
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Neuro.Nodes.Base, proto: Cuda.Graph

    def __graph__(graph) do
      graph |> chain(:padding, Padding, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: Padding
  end

  describe "padding node" do
    setup do
      log_level = Logger.level()
      Logger.configure(level: :error)
      on_exit(fn -> Logger.configure(level: log_level) end)
    end

    test "simple padding" do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.7,
           1.0, 0.1, 0.2]

      opts = [network: Wrapper,
              network_options: [size: {3, 3}, padding_size: {1, 1}, padding: 1.0]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output |> round!() == [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.1, 0.2, 0.3, 1.0],
        [1.0, 0.5, 0.6, 0.7, 1.0],
        [1.0, 1.0, 0.1, 0.2, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
      ]
    end
  end
end
