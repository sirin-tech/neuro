defmodule Mix.Tasks.Neuro.Benchmark do
  alias Neuro.Layers.Convolution
  alias Neuro.Layers.FullyConnected
  alias Neuro.Network

  defmodule MNISTNetwork do
    use Network

    input  {28, 28}
    output 10

    def graph(graph) do
      graph
      |> chain(:conv1, Convolution, kernel_size: {5, 5, 20}, pooling: 2)
      |> chain(:conv2, Convolution, kernel_size: {5, 5, 50}, pooling: 2)
      |> chain(:fc1,   FullyConnected, out_size: 500)
      |> chain(:fc2,   FullyConnected, out_size: 10)
      |> close()
    end
  end

  defp random_data(num) do
    for _ <- 1..num, do: :rand.uniform()
  end

  def run(_) do
    :rand.seed(:exs64)
    weights = %{
      conv1: random_data(5 * 5 * 20),
      conv2: random_data(5 * 5 * 50),
      fc1:   random_data(50 * 4 * 4 * 500),
      fc2:   random_data(500 * 10)
    }
    biases = %{
      conv1: random_data(5 * 5 * 20),
      conv2: random_data(5 * 5 * 50),
      fc1:   random_data(500),
      fc2:   random_data(10)
    }

    {:ok, _} = MNISTNetwork.start_link(shared: %{weights: weights, biases: biases})
    i = random_data(28 * 28)

    n = 200
    start = System.monotonic_time(:microseconds)
    1..n |> Enum.each(fn _ ->
      MNISTNetwork.run(%{input: i})
    end)
    finish = System.monotonic_time(:microseconds)
    total = Float.round((finish - start) / 1_000_000, 3)
    IO.puts("Run #{n} iterations. Elapsed time: #{total} seconds")
  end
end
