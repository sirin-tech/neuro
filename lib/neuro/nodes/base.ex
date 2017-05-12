defmodule Neuro.Nodes.Base do
  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.GPUNode
      import unquote(__MODULE__)

      def vars(_), do: %{}

      def __assigns__(opts, _env) do
        %{vars: vars(opts)}
      end

      def __pins__(assigns) do
        [input(:input, input_type(assigns.vars)),
         output(:output, output_type(assigns.vars))]
      end

      defoverridable __assigns__: 2, __pins__: 1, vars: 1
    end
  end

  def input_type(%{z: 1, x: x, y: y, f: f}), do: {{f, x}, {f, y}}
  def input_type(%{z: z, x: x, y: y, f: f}), do: {{f, x}, {f, y}, {f, z}}

  def output_type(%{oz: 1, ox: x, oy: y, f: f}), do: {{f, x}, {f, y}}
  def output_type(%{oz: z, ox: x, oy: y, f: f}), do: {{f, x}, {f, y}, {f, z}}

  def triple_size({x, y}), do: {x, y, 1}
  def triple_size({_, _, _} = tuple), do: tuple
  def triple_size(x) when is_integer(x), do: {x, x, 1}
  def triple_size(_) do
    raise CompileError, description: "Invalid size specified"
  end

  def stride({_, _} = tuple), do: tuple
  def stride(x) when is_integer(x), do: {x, x}
  def stride(_), do: {1, 1}

  def float_size(x) when x in [16, 32, 64], do: x / 8
  def float_size(_), do: 4

  def input(assigns) do
    f = assigns.vars.f
    x = assigns.vars.x
    y = assigns.vars.y
    case assigns.vars.z do
      1 -> {{f, x}, {f, y}}
      z -> {{f, x}, {f, y}, {f, z}}
    end
  end
end
