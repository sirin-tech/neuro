defmodule Neuro.Worker do
  use GenServer

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    with {:ok, module} <- Keyword.fetch(opts, :network),
         {:ok, vars} <- Keyword.fetch(opts, :vars),
         {:ok, cuda} <- Cuda.start_link() do
      st = %{cuda: cuda, vars: vars, module: module}
      load_network(st)
    end
  end

  defp load_network(st) do
    {:ok, st}
  end
end
