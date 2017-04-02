defmodule Neuro.Convolution do
  use Neuro.Layer

  @ptx "lib/neuro/ptx/convolution.ptx"
  @func "convolution"

  def run({ix, iy, input}, {wx, wy, wz, w}, {sx, sy}) do
    {:ok, cuda} = Cuda.start_link(port_bin: "../cuda/priv/cuda_port")
    nx = round((ix - wx + sx) / sx)
    ny = round((iy - wy + sy) / sy)
    size = nx * ny * wz * 32
    {:ok, moutput} = Cuda.memory_load(cuda, <<0::size(size)>>)
    {:ok, minput}  = Cuda.memory_load(cuda, input)
    {:ok, mw}      = Cuda.memory_load(cuda, w)
    {:ok, module} = Cuda.compile(cuda, {:file, @ptx})
    # IO.inspect({nx, ny, wz})
    :ok = Cuda.run(cuda, module, @func, {nx, ny, wz}, [minput, mw, moutput, wx, wy, sx, sy])
    {:ok, out} = Cuda.memory_read(cuda, moutput)
    for <<x::float-little-32 <- out>>, do: x
  end
end
