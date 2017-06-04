defmodule Neuro.Nodes.FullyConnected do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{back_propagation: true, vars: vars}}) do
    [{:run, {"back", vars.block, vars.grid, [:shared]}}]
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

      <%= if training?() do %>
        mad.lo.u64  %state_ptr, %x, <%= var(:float_size) %>, %b_ptr;
        <%= if shared_offset(:states) > 0 do %>
          add.u64   %state_ptr, %state_ptr, <%= shared_offset(:states) %>;
        <% end %>
      <% end %>

      // biases.offset = biases + tid.x * float_size
      mad.lo.u64    %b_ptr, %x, <%= var(:float_size) %>, %b_ptr;
      <%= if shared_offset(:biases) > 0 do %>
        add.u64     %b_ptr, %b_ptr, <%= shared_offset(:biases) %>;
      <% end %>
      ld.global.<%= var(:f) %> %b, [%b_ptr];

      // (%o_ptr) output.offset = output + tid.x * float_size + output.offset
      mad.lo.u64    %o_ptr, %x, <%= var(:float_size) %>, %i_ptr;
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
      .reg .u64 %loss_ptr, %inference_ptr, %state_ptr, %output_ptr, %w_ptr, %speed_ptr, %count, %x;
      .reg .<%= var(:f) %> %activation, %loss, %w, %acc, %speed, %tmp;
      .reg .pred p;

      ld.param.u64 	%loss_ptr, [pins];
    	ld.param.u64 	%speed_ptr, [shared];
      cvt.u64.u32   %x, %ctaid.x;

      mad.lo.u64   %output_ptr, %x, <%= var(:float_size) %>, %loss_ptr;
      <%= if offset(:pins, :input) > 0 do %>
        add.u64    %output_ptr, %output_ptr, <%= offset(:pins, :input) %>;
      <% end %>
      mad.lo.u64   %inference_ptr, %x, <%= var(:float_size) %>, %loss_ptr;
      <%= if offset(:pins, :inference) > 0 do %>
        add.u64    %inference_ptr, %inference_ptr, <%= offset(:pins, :inference) %>;
      <% end %>

      <%= if offset(:pins, :output) > 0 do %>
        add.u64    %loss_ptr, %loss_ptr, <%= offset(:pins, :output) %>;
      <% end %>

      <%= if shared_offset(:states) > 0 do %>
        add.u64    %state_ptr, %speed_ptr, <%= shared_offset(:states) %>;
      <% end %>
      mad.lo.u64    %w_ptr, %x, <%= var(:float_size) %>, %speed_ptr;
      <%= if shared_offset(:weights) > 0 do %>
        add.u64     %w_ptr, %w_ptr, <%= shared_offset(:weights) %>;
      <% end %>
      <%= if offset(:shared, :speed) > 0 do %>
        add.u64     %speed_ptr, %speed_ptr, <%= offset(:shared, :speed) %>;
      <% end %>

      mov.u64                  %count, 0;
      mov.<%= var(:f) %>       %acc, 0.0;
    loss_loop:
      // load variables
      ld.global.<%= var(:f) %> %activation, [%state_ptr];
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      ld.global.<%= var(:f) %> %w, [%w_ptr];
      <%= include ctx, var(:activation), :back_propagation, in: "activation", pred: "p", f: var(:f) %>

      // calculate loss for previous layer
      mul.rn.<%= var(:f) %>    %tmp, %w, %activation;
      mad.rn.<%= var(:f) %>    %acc, %loss, %tmp, %acc;

      add.u64                  %count, %count, 1;
      setp.lo.u64              p, %count, <%= var(:ox) %>;
      @p add.u64               %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      @p add.u64               %state_ptr, %state_ptr, <%= var(:float_size) %>;
      @p add.u64               %w_ptr, %w_ptr, <%= var(:x) * var(:float_size) %>;
      @p bra                   loss_loop;

      st.global.<%= var(:f) %> [%output_ptr], %acc;

      // calculate weights delta
      ld.global.<%= var(:f) %> %tmp, [%inference_ptr];
      ld.global.<%= var(:f) %> %speed, [%speed_ptr];
    weight_loop:
      mul.rn.<%= var(:f) %>    %acc, %tmp, %activation;
      mul.rn.<%= var(:f) %>    %acc, %acc, %loss;
      mul.rn.<%= var(:f) %>    %acc, %acc, %speed;
      sub.<%= var(:f) %>       %acc, %w, %acc;
      st.global.<%= var(:f) %> [%w_ptr], %acc;

      sub.u64                  %count, %count, 1;
      setp.eq.u64              p, %count, 0;
      @p ret;

      sub.u64                  %loss_ptr, %loss_ptr, <%= var(:float_size) %>;
      sub.u64                  %state_ptr, %state_ptr, <%= var(:float_size) %>;
      sub.u64                  %w_ptr, %w_ptr, <%= var(:x) * var(:float_size) %>;
      ld.global.<%= var(:f) %> %activation, [%state_ptr];
      ld.global.<%= var(:f) %> %loss, [%loss_ptr];
      ld.global.<%= var(:f) %> %w, [%w_ptr];
      <%= include ctx, var(:activation), :back_propagation, in: "activation", pred: "p", f: var(:f) %>
      bra                      weight_loop;
    <% end %>
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
      {x, 1, 1}
    else
      {ox, 1, 1}
    end

    %{x: x, y: 1, z: 1,
      ox: ox, oy: 1, oz: 1,
      activation: activation,
      block: block, grid: grid}
  end

  def shared(key, vars) do
    shared = %{weights: %{key => {vars.f, vars.x * vars.ox}},
               biases:  %{key => {vars.f, vars.ox}}}
    shared = if vars.training do
      Map.merge(shared, %{states: %{key => {vars.f, vars.ox}}})
    else
      shared
    end
    %{shared: shared}
  end
end
