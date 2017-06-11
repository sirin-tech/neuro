defmodule Neuro.Nodes.FullyConnected do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    {grid, bgrid} = vars.grid
    [{:run, {"back", vars.block, grid, [:shared]}},
     {:run, {"back_biases", vars.block, bgrid, [:shared]}}]
  end
  def __batch__(%{assigns: %{vars: vars}}) do
    [{:run, {"inference", vars.block, vars.grid, [:shared]}}]
  end

  def __ptx__(%{assigns: %{back_propagation: true}}) do
    back_ptx()
  end
  def __ptx__(_node) do
    inference_ptx()
  end

  def inference_ptx() do
    """
    <%= defkernel ctx, "inference", shared: u64.ptr do %>
      .reg .u64            %w_ptr, %b_ptr, %i_ptr, %o_ptr, %x, %last;
    	.reg .<%= var(:f) %> %f<4>, %acc, %b, %i, %w;
    	.reg .pred	         p;
      <%= if training?() do %>
        .reg .u64          %state_ptr;
      <% end %>

      ld.param.u64  %i_ptr, [pins];
    	ld.param.u64 	%b_ptr, [shared];

      cvt.u64.u32   %x, %ctaid.x;

      // weight.offset = weight + tid.x * ix * float_size
      mad.lo.u64    %w_ptr, %x, <%= var(:x) * var(:float_size) %>, %b_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>

      mul.lo.u64    %x, %x, <%= var(:float_size) %>;

      <%= if training?() do %>
        add.u64     %state_ptr, %x, %b_ptr;
        <%= if shared_offset(:states) > 0 do %>
          add.u64   %state_ptr, %state_ptr, <%= shared_offset(:states) %>;
        <% end %>
      <% end %>

      // biases.offset = biases + tid.x * float_size
      add.u64       %b_ptr, %x, %b_ptr;
      <%= if shared_offset(:biases) > 0 do %>
        add.u64     %b_ptr, %b_ptr, <%= shared_offset(:biases) %>;
      <% end %>
      ld.global.<%= var(:f) %> %b, [%b_ptr];

      // (%o_ptr) output.offset = output + tid.x * float_size + output.offset
      add.u64       %o_ptr, %x, %i_ptr;
      <%= if offset(:pins, :output) > 0 do %>
        add.u64     %o_ptr, %o_ptr, <%= offset(:pins, :output) %>;
      <% end %>

      // (%i_ptr) input.offset = pins + input.offset
      <%= if offset(:pins, :input) > 0 do %>
        add.u64     %i_ptr, %i_ptr, <%= offset(:pins, :input) %>;
      <% end %>

      // %acc - accumulator
      mov.<%= var(:f) %>       %acc, 0.0;
      add.u64                  %last, %i_ptr, <%= var(:x) * var(:float_size) %>;
    loop_x:
      ld.global.<%= var(:f) %> %i, [%i_ptr];
      ld.global.<%= var(:f) %> %w, [%w_ptr];
      // accumulator = accumulator + w[x] * input[x]
      mad.rn.<%= var(:f) %>    %acc, %i, %w, %acc;
      // next values
      add.u64                  %i_ptr, %i_ptr, <%= var(:float_size) %>;
      add.u64                  %w_ptr, %w_ptr, <%= var(:float_size) %>;
      // count x
      setp.lo.u64              p, %i_ptr, %last;
      @p bra                   loop_x;

      // bias
      add.f32 %acc, %acc, %b;

      <%= if training?() do %>
        st.global.<%= var(:f) %> [%state_ptr], %acc;
      <% end %>

      // relu activation
      <%= include ctx, var(:activation), in: "acc", pred: "p", f: var(:f) %>
      st.global.<%= var(:f) %> [%o_ptr], %acc;
      ret;
    <% end %>
    """
  end

  defp back_ptx() do
    """
    <%= defkernel ctx, "back", shared: u64.ptr do %>
      .reg .u64 %loss_ptr, %inference_ptr, %state_ptr, %output_ptr, %w_ptr, %dw_ptr, %count, %x;
      .reg .<%= var(:f) %> %activation, %loss, %w, %acc, %inference, %tmp;
      .reg .pred p;

      ld.param.u64 	%loss_ptr, [pins];
    	ld.param.u64 	%dw_ptr, [shared];
      cvt.u64.u32   %x, %ctaid.x;

      mul.lo.u64    %x, %x, <%= var(:float_size) %>;

      add.u64       %output_ptr, %x, %loss_ptr;
      <%= if offset(:pins, :input) > 0 do %>
        add.u64     %output_ptr, %output_ptr, <%= offset(:pins, :input) %>;
      <% end %>
      add.u64       %inference_ptr, %x, %loss_ptr;
      <%= if offset(:pins, :inference) > 0 do %>
        add.u64     %inference_ptr, %inference_ptr, <%= offset(:pins, :inference) %>;
      <% end %>

      <%= if offset(:pins, :output) > 0 do %>
        add.u64     %loss_ptr, %loss_ptr, <%= offset(:pins, :output) %>;
      <% end %>

      <%= if shared_offset(:states) > 0 do %>
        add.u64     %state_ptr, %dw_ptr, <%= shared_offset(:states) %>;
      <% end %>
      add.u64       %w_ptr, %x, %dw_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>
      add.u64       %dw_ptr, %x, %dw_ptr;
      <%= if offset(:shared, :dw) > 0 do %>
        add.u64     %dw_ptr, %dw_ptr, <%= offset(:shared, :dw) %>;
      <% end %>

      mov.u64                  %count, 0;
      mov.<%= var(:f) %>       %acc, 0.0;
      ld.global.<%= var(:f) %> %inference, [%inference_ptr];

    loss_loop:
      // load variables
      ld.global.<%= var(:f) %> %activation, [%state_ptr];
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      ld.global.<%= var(:f) %> %w, [%w_ptr];
      <%= include ctx, var(:activation), :back_propagation, in: "activation", pred: "p", f: var(:f) %>

      // calculate loss for previous layer
      mul.rn.<%= var(:f) %>    %tmp, %w, %activation;
      mad.rn.<%= var(:f) %>    %acc, %loss, %tmp, %acc;

      // calculate weight delta
      ld.global.<%= var(:f) %> %tmp, [%dw_ptr];
      mad.rn.<%= var(:f) %>    %tmp, %inference, %loss, %tmp;
      st.global.<%= var(:f) %> [%dw_ptr], %tmp;

      add.u64                  %count, %count, 1;
      setp.lo.u64              p, %count, <%= var(:ox) %>;
      @p add.u64               %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      @p add.u64               %state_ptr, %state_ptr, <%= var(:float_size) %>;
      @p add.u64               %w_ptr, %w_ptr, <%= var(:x) * var(:float_size) %>;
      @p add.u64               %dw_ptr, %dw_ptr, <%= var(:x) * var(:float_size) %>;
      @p bra                   loss_loop;

      st.global.<%= var(:f) %> [%output_ptr], %acc;

      ret;
    <% end %>

    <%= defkernel ctx, "back_biases", shared: u64.ptr do %>
      .reg .u64 %db_ptr, %loss_ptr, %x;
      .reg .<%= var(:f) %> %db, %loss;

    	ld.param.u64 	%db_ptr, [shared];
    	ld.param.u64 	%loss_ptr, [pins];
      cvt.u64.u32   %x, %ctaid.x;
      mul.lo.u64    %x, %x, <%= var(:float_size) %>;

      add.u64       %loss_ptr, %x, %loss_ptr;
      <%= if offset(:pins, :output) > 0 do %>
        add.u64     %loss_ptr, %loss_ptr, <%= offset(:pins, :output) %>;
      <% end %>

      add.u64       %db_ptr, %x, %db_ptr;
      <%= if offset(:shared, :db) > 0 do %>
        add.u64     %db_ptr, %db_ptr, <%= offset(:shared, :db) %>;
      <% end %>

      // calculate bias delta
      ld.global.<%= var(:f) %> %db, [%db_ptr];
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      add.rn.<%= var(:f) %>    %db, %db, %loss;
      st.global.<%= var(:f) %> [%db_ptr], %db;

      ret;
    <%= end %>
    """
  end

  def vars(opts, %{gpu_info: info}) do
    x              = opts |> Keyword.get(:size) |> Base.plane_size()
    ox             = opts |> Keyword.get(:out_size) |> Base.plane_size()
    activation     = opts |> Keyword.get(:activation, :relu) |> Base.activation()

    {max_x, _, _} = info[:max_block]
    if ox > max_x do
      raise RuntimeError, message: "Maximum allowed layer size is #{max_x}"
    end
    block = {1, 1, 1}
    grid = if opts[:back_propagation] do
      {{x, 1, 1}, {ox, 1, 1}}
    else
      {ox, 1, 1}
    end

    %{x: x, y: 1, z: 1,
      ox: ox, oy: 1, oz: 1,
      activation: activation,
      block: block, grid: grid}
  end

  def shared(key, vars) do
    w_size = vars.x * vars.ox
    b_size = vars.ox
    shared = %{weights: %{key => {vars.f, w_size}},
               biases:  %{key => {vars.f, b_size}}}
    shared = if vars.training do
      Map.merge(shared, %{states: %{key => {vars.f, vars.ox}}})
    else
      shared
    end
    shared = if vars.back_propagation do
      Map.merge(shared, %{dw: %{key => {vars.f, w_size}},
                          db: %{key => {vars.f, b_size}}})
    else
      shared
    end
    %{shared: shared}
  end
end
