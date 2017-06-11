defmodule Neuro.Nodes.Terminator do
  use Cuda.Graph.GPUNode

  def __assigns__(_id, opts, _env) do
    size = Neuro.Nodes.Base.plane_size(Keyword.fetch!(opts, :size))
    group = Keyword.get(opts, :group, nil)
    %{vars: %{size: size, group: group}}
  end

  def __pins__(%{vars: %{size: size, group: group}, env: env}) do
    type = {Cuda.Env.f(env), size}
    [pin(:input, :terminator, type, group)]
  end
end
