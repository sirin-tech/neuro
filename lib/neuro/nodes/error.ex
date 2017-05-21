defmodule Neuro.Nodes.Error do
  use Cuda.Graph.GPUNode

  def __pins__(%{options: opts}) do
    input = Keyword.get(opts, :input)
    output = Keyword.get(opts, :output)
    input = %{input | id: :input, type: :input}
    reply = %{input | id: :reply, type: :input}
    output = %{output | id: :output, type: :output}
    [input, reply, output]
  end

  #def __ptx__(_) do
  #  ""
  #end
end
