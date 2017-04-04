defmodule Neuro.FullConnected do
  use Neuro.Layer

  @ptx "lib/neuro/ptx/fullconnected.ptx"
  @func "fullconnected"

  def run({input, weights, ix}, ox) when is_list(input) do
    run({to_arr(input), weights, ix}, ox)
  end
  def run({input, weights, ix}, ox) when is_list(weights) do
    run({input, to_arr(weights), ix}, ox)
  end
  def run({input, weights, ix}, ox) do
    {:ok, cuda} = Cuda.start_link(port_bin: "../cuda/priv/cuda_port")
    size = ox * 32
    {:ok, moutput} = Cuda.memory_load(cuda, <<0::size(size)>>)
    {:ok, minput}  = Cuda.memory_load(cuda, input)
    {:ok, mw}      = Cuda.memory_load(cuda, weights)
    {:ok, module} = Cuda.compile(cuda, {:file, @ptx})
    :ok = Cuda.run(cuda, module, @func, ox, [minput, mw, moutput, ix])
    {:ok, out} = Cuda.memory_read(cuda, moutput)
    for <<x::float-little-32 <- out>>, do: x
  end

  def to_arr([val]), do: <<val::float-little-32>>
  def to_arr([val | rest]) do
    <<val::float-little-32>> <> to_arr(rest)
  end
end
