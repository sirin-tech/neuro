defmodule Neuro.Test.NodesHelpers do
  import Neuro.Test.CudaHelpers
  alias Cuda.Compiler.Unit
  alias Cuda.Runner

  defp make_binary(v) when is_list(v) do
    v |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
  end

  defp round!(l) when is_list(l), do: Enum.map(l, &round!/1)
  defp round!(f) when is_float(f), do: Float.round(f, 1)
  defp round!(x), do: x

  def run(node, inputs, args \\ %{}) do
    args = args |> Enum.map(fn
      {k, v} -> {k, v |> make_binary}
      x      -> x
    end)

    l = Logger.level()
    Logger.configure(level: :error)
    {:ok, node} = Unit.compile(node, context())
    Logger.configure(level: l)

    {:ok, cuda} = Cuda.start_link()
    args = Enum.reduce(args, %{}, fn
      {k, v}, acc ->
        with {:ok, m} <- Cuda.memory_load(cuda, v) do
          Map.put(acc, k, m)
        else
          _ -> acc
        end
      _, acc ->
        acc
    end)

    opts = [cuda: cuda, args: args]
    with {:ok, outputs} <- Runner.run(node, inputs, opts) do
      outputs
      |> Enum.map(fn {k, v} -> {k, round!(v)} end)
      |> Enum.into(%{})
    else
      _ -> %{}
    end
  end
end
