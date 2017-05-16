defmodule Neuro.Nodes.FullyConnected do
  alias Neuro.Nodes.Base
  use Base

  def __batch__(%{assigns: %{vars: vars}}) do
    [{"fully_connected", {vars.ox, vars.oy, vars.oz}, {1, 1, 1}, [:w, :b]}]
  end

  def __ptx__(_node) do
    """
    .version 4.3
    .target sm_20
    .address_size 64

    <%= defkernel ctx, "fully_connected", weights: u64.ptr, biases: u64.ptr do %>
      .reg .u64   %cd<4>;
      .reg .u32   %c<1>;
    	.reg .u32		%v<1>;
    	.reg .<%= var(ctx, :f) %> %f<4>;
    	.reg .pred	p;

      ld.param.u64 	%cd0, [pins];
    	ld.param.u64 	%cd1, [weights];
      ld.param.u64  %cd3, [biases];
    	mov.u32				%c0, %tid.x;

      // (%cd1) weight.offset = weight + tid.x * ix * float_size
      mad.wide.u32 %cd1, %c0, <%= var(ctx, :x) * var(ctx, :float_size) %>, %cd1;
      <%= if var(ctx, :weights_offset) > 0 do %>
        add.u64    %cd1, %cd1, <%= var(ctx, :weights_offset) %>;
      <% end %>

      // (%cd3) biases.offset = biases + tid.x * float_size
      mad.wide.u32 %cd3, %c0, <%= var(ctx, :float_size) %>, %cd3;
      <%= if var(ctx, :biases_offset) > 0 do %>
        add.u64    %cd3, %cd3, <%= var(ctx, :biases_offset) %>;
      <% end %>
      ld.global.<%= var(ctx, :f) %> %f3, [%cd3];

      // (%cd2) output.offset = output + tid.x * float_size + output.offset
      mad.wide.u32 %cd2, %c0, <%= var(ctx, :float_size) %>, %cd0;
      <%= if offset(ctx, :output) > 0 do %>
        add.u64    %cd2, %cd2, <%= offset(ctx, :output) %>;
      <% end %>

      // (%cd0) input.offset = pins + input.offset
      <%= if offset(ctx, :input) > 0 do %>
        add.u64    %cd0, %cd0, <%= offset(ctx, :input) %>;
      <% end %>

      // %f0 - accumulator
      // %v0 - x counter
      mov.<%= var(ctx, :f) %>       %f0, 0.0;
      mov.u32                       %v0, <%= var(ctx, :x) %>;
    loop_x:
      ld.global.<%= var(ctx, :f) %> %f1, [%cd0];
      ld.global.<%= var(ctx, :f) %> %f2, [%cd1];
      // accumulator = accumulator + w[x] * input[x]
      mad.rn.<%= var(ctx, :f) %>    %f0, %f1, %f2, %f0;
      // next values
      add.u64                       %cd0, %cd0, <%= var(ctx, :float_size) %>;
      add.u64                       %cd1, %cd1, <%= var(ctx, :float_size) %>;
      // count x
      sub.u32                       %v0, %v0, 1;
      setp.ne.u32                   p, %v0, 0;
      @p bra                        loop_x;

      // bias
      add.f32 %f0, %f0, %f3;

      // relu activation
      <%= include ctx, var(ctx, :activation), in: "f0", pred: "p" %>
      st.global.<%= var(ctx, :f) %> [%cd2], %f0;
      ret;
    <% end %>
    """
  end

  def vars(opts) do
    x              = opts |> Keyword.get(:size) |> Base.plane_size()
    ox             = opts |> Keyword.get(:out_size) |> Base.plane_size()
    float_size     = opts |> Keyword.get(:float_size) |> Base.float_size()
    activation     = opts |> Keyword.get(:activation, :relu) |> Base.activation()
    weights_offset = opts |> Keyword.get(:weights_offset, 0)
    biases_offset  = opts |> Keyword.get(:biases_offset, 0)
    f = "f#{float_size * 8}"

    %{x: x, y: 1, z: 1,
      ox: ox, oy: 1, oz: 1,
      activation: activation,
      f: f, float_size: float_size,
      weights_offset: weights_offset, biases_offset: biases_offset}
  end
end
