defmodule Neuro.Test.NodesHelpers do
  import Neuro.Test.CudaHelpers
  alias Cuda.Compiler.Unit
  alias Cuda.Graph.Factory
  alias Cuda.Runner

  defp make_binary(v) when is_list(v) do
    v |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
  end

  def round!(l) when is_list(l), do: Enum.map(l, &round!/1)
  def round!(f) when is_float(f), do: Float.round(f, 1)
  def round!(x), do: x

  def run(proto, id, module, opts, inputs, args \\ %{}) do
    args = args |> Enum.map(fn
      {k, v} -> {k, v |> make_binary}
      x      -> x
    end)

    l = Logger.level()
    Logger.configure(level: :error)

    {:ok, cuda} = Cuda.start_link()
    {:ok, info} = Cuda.device_info(cuda)

    node = Factory.new(proto, id, module, opts, %Cuda.Env{gpu_info: info})

    {:ok, node} = Unit.compile(node, context())
    Logger.configure(level: l)

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

    with {:ok, node}    <- Runner.load(node, cuda: cuda),
         {:ok, outputs} <- Runner.run(node, inputs, cuda: cuda, args: args) do
      outputs
      |> Enum.map(fn {k, v} -> {k, round!(v)} end)
      |> Enum.into(%{})
    else
      _ -> %{}
    end
  end
end
