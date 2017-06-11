defmodule Neuro.Stream do
  @moduledoc """
  Represents manageable data stream between producer and consumer
  """
  use Supervisor

  @doc """
  Starts general supervisor, accepts the following as a parameters:
    consumer    - process pid that realise Consumer startegy
    module      - module that relise Producer stratehy
    module_opts - initial options for Producer module
    arity       - represents data type and it's structure
    name        - optional parameter, give a name to supervisor process
  """
  @spec start_link(opts :: [keyword]) :: {:ok, pid}
  def start_link(opts) do
    case Keyword.get(opts, :name) do
      nil  -> Supervisor.start_link(__MODULE__, opts)
      name -> Supervisor.start_link(__MODULE__, opts, name: name)
    end
  end

  def init(opts) do
    with cpid    when is_pid(cpid)       <- Keyword.get(opts, :consumer),
         rmodule when is_atom(rmodule)   <- Keyword.get(opts, :module),
         ropts   when is_list(ropts)     <- Keyword.get(opts, :module_opts),
         darity  when not is_nil(darity) <- Keyword.get(opts, :arity) do

      workers = [
       worker(rmodule, [{ropts, cpid, darity}], id: :reader, restart: :transient)
      ]

      GenStage.cast(cpid, {:stream_register, self()})

      supervise(workers, strategy: :one_for_one)
    else
      _ -> {:error, :badargs}
    end
  end

  def read(listener_pid, num) do
    GenStage.cast(listener_pid, {:ask, num})
  end

  @doc """
  Close Producer child and Supervisor processes
  """
  @spec close(supervisor_pid :: pid) :: :ok
  def close(visor_pid) do
    GenStage.cast(child(visor_pid, :reader), :exit)
    Supervisor.stop(visor_pid, :normal)
  end

  defp child(visor_pid, id) do
    visor_pid
    |> Supervisor.which_children()
    |>  Enum.reduce(nil, fn
      {^id, pid, _, _}, _ -> pid
      _, acc              -> acc
    end)
  end
end
