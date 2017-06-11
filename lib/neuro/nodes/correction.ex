defmodule Neuro.Nodes.Correction do
  use Cuda.Graph.GPUNode

  def __assigns__(_id, opts, env) do
    overrides = opts |> Keyword.take(~w(f float_size)a) |> Enum.into(%{})
    overrides = env
                |> Map.take(~w(float_size)a)
                |> Map.put(:f, Cuda.Env.f(env))
                |> Map.merge(overrides)

    vars = opts
           |> Keyword.merge(overrides |> Enum.into([]))
           |> vars(env)
           |> Enum.into(%{})
           |> Map.merge(overrides)

    %{vars: vars}
  end

  def __pins__(%{vars: %{input: input}}) do
    [consumer(:input, input)]
  end

  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"weights", {1, 1, 1}, {1, 1, vars.weights}, [:shared, :speed]}},
     {:run, {"biases", {1, 1, 1}, {1, 1, vars.biases}, [:shared, :speed]}}]
  end

  def __ptx__(%{assigns: %{vars: %{f: f}}}) do
    """
    <%= defkernel ctx, "weights", shared: u64.ptr, speed: #{f} do %>
      .reg .u64 %x, %w_ptr, %dw_ptr;
      .reg .<%= var(:f) %> %w, %dw, %speed;

    	ld.param.u64 	%w_ptr, [shared];
      ld.param.<%= var(:f) %> %speed, [speed];
      cvt.u64.u32   %x, %ctaid.z;

      mad.lo.u64    %w_ptr, %x, <%= var(:float_size) %>, %w_ptr;
      <%= if offset(:shared, :dw) > 0 do %>
        add.u64     %dw_ptr, %w_ptr, <%= offset(:shared, :dw) %>;
      <% end %>
      <%= if offset(:shared, :weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= offset(:shared, :weights) %>;
      <% end %>
      ld.global.<%= var(:f) %>  %w, [%w_ptr];
      ld.global.<%= var(:f) %>  %dw, [%dw_ptr];
      mad.rn.<%= var(:f) %>     %w, %dw, %speed, %w;
      st.global.<%= var(:f) %>  [%w_ptr], %w;
      ret;
    <% end %>

    <%= defkernel ctx, "biases", shared: u64.ptr, speed: #{f} do %>
      .reg .u64 %x, %b_ptr, %db_ptr;
      .reg .<%= var(:f) %> %b, %db, %speed;

    	ld.param.u64 	%b_ptr, [shared];
      ld.param.<%= var(:f) %> %speed, [speed];
      cvt.u64.u32   %x, %ctaid.z;

      mad.lo.u64    %b_ptr, %x, <%= var(:float_size) %>, %b_ptr;
      <%= if offset(:shared, :db) > 0 do %>
        add.u64     %db_ptr, %b_ptr, <%= offset(:shared, :db) %>;
      <% end %>
      <%= if offset(:shared, :biases) > 0 do %>
        add.u64     %b_ptr, %b_ptr, <%= offset(:shared, :biases) %>;
      <% end %>
      ld.global.<%= var(:f) %>  %b, [%b_ptr];
      ld.global.<%= var(:f) %>  %db, [%db_ptr];
      mad.rn.<%= var(:f) %>     %b, %db, %speed, %b;
      st.global.<%= var(:f) %>  [%b_ptr], %b;
      ret;
    <% end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    weights   = opts |> Keyword.get(:weights, 0)
    biases    = opts |> Keyword.get(:biases, 0)
    input     = opts |> Keyword.get(:input, 0)
    #{x, y, z} = opts |> Keyword.get(:size) |> Base.triple_size()

    {max_x, _, _} = info[:max_block]
    if weights > max_x do
      raise RuntimeError, message: "Maximum allowed weights number is #{max_x}"
    end
    if biases > max_x do
      raise RuntimeError, message: "Maximum allowed biases number is #{max_x}"
    end

    %{weights: weights, biases: biases, input: input}
  end

  #def shared(key, vars) do
  #  %{shared: %{dw: %{key => {vars.f, vars.weights}},
  #              db: %{key => {vars.f, vars.biases}}}}
  #end
end
