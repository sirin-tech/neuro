defmodule Neuro.Network do
  @layers __MODULE__.Layers

  defmacro __using__(_opts) do
    quote do
      use GenServer
      import unquote(__MODULE__), only: [layer: 2, layer: 3]
      @before_compile unquote(__MODULE__)
      Module.register_attribute(__MODULE__, unquote(@layers), accumulate: true)

      def start_link(opts \\ []) do
        {name, opts} = Keyword.pop(opts, :name, __MODULE__)
        GenServer.start_link(__MODULE__, opts, name: name)
      end

      def init(_opts) do
        {:ok, %{}}
      end
    end
  end

  defmacro __before_compile__(env) do
    layers = env |> Module.get_attribute(@layers)
    quote do
      def __layers__ do
        unquote(layers)
      end
    end
  end

  defmacro layer(name, module, opts \\ []) do
    quote do
      Module.put_attribute(unquote(@layers), unquote(name),
                           {unquote(module), unquote(opts)})
    end
  end
end
