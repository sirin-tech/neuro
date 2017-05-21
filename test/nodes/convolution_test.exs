defmodule Neuro.Nodes.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Convolution
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Neuro.Nodes.Base, proto: Cuda.Graph

    def shared(vars) do
      %{weights: {vars.f, vars.wx * vars.wy * vars.wz},
        biases:  {vars.f, vars.wx * vars.wy * vars.wz}}
    end

    def __graph__(graph) do
      opts = Keyword.put(graph.assigns.options, :alias, :network)
      graph |> chain(:conv, Convolution, opts) |> close()
    end

    defdelegate vars(opts, env), to: Convolution
  end

  describe "convolution node" do
    setup do
      {:ok, shared} = Cuda.Shared.start_link()
      log_level = Logger.level()
      Logger.configure(level: :warn)
      on_exit(fn -> Logger.configure(level: log_level) end)
      [shared: shared]
    end

    test "simple convolution", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]
      b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      opts = [network: Wrapper,
              shared_pid: ctx[:shared],
              shared: %{weights: %{network: w}, biases: %{network: b}},
              network_options: [size: {4, 4}, kernel_size: {2, 2, 2}]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output |> round!() == [
        [[4.4, 5.4, 6.4],
         [5.1, 3.1, 4.1],
         [4.4, 4.4, 5.4]],

        [[10.0, 12.6, 15.2],
         [13.9,  9.5, 12.1],
         [12.4, 10.0, 12.6]]
      ]
    end

    test "relu activation", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]
      w = [[1.0, 2.0, 3.0, 4.0],
           [-5.0, -6.0, -7.0, -8.0]]
      b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      opts = [network: Wrapper,
              shared_pid: ctx[:shared],
              shared: %{weights: %{network: w}, biases: %{network: b}},
              network_options: [size: {4, 4}, kernel_size: {2, 2, 2}]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        [[4.4, 5.4, 6.4],
         [5.1, 3.1, 4.1],
         [4.4, 4.4, 5.4]],

        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]
      ]
    end
  end
end
